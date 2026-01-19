"""
Cog predictor for TRELLIS2 - Image to 3D Generation
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import tempfile
from pathlib import Path
from typing import Optional

import cv2
import imageio
import torch
from cog import BasePredictor, Input, Path as CogPath
from PIL import Image
from huggingface_hub import login as hf_login

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory"""
        # Check for pre-downloaded models (used in Replicate deployment)
        local_model_path = "/src/models/TRELLIS.2-4B"
        
        # Authenticate with HuggingFace if token provided (for local testing)
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            print("Authenticating with HuggingFace...")
            hf_login(token=hf_token)
        
        # Load from local path if available, otherwise download from HuggingFace
        print("Loading TRELLIS2 pipeline...")
        if os.path.exists(local_model_path):
            print(f"Using pre-downloaded model from {local_model_path}")
            self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(local_model_path)
        else:
            print("Downloading model from HuggingFace...")
            self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        self.pipeline.cuda()
        
        # Load default environment map
        print("Loading environment map...")
        envmap_path = "assets/hdri/forest.exr"
        self.envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        ))
        print("Setup complete!")

    def predict(
        self,
        image: CogPath = Input(
            description="Input image for 3D generation"
        ),
        output_format: str = Input(
            description="Output format: 'glb' for 3D model, 'video' for visualization, or 'both'",
            default="both",
            choices=["glb", "video", "both"]
        ),
        resolution: int = Input(
            description="Output resolution (512, 1024, or 1536)",
            default=512,
            choices=[512, 1024, 1536]
        ),
        simplify_mesh: bool = Input(
            description="Simplify the mesh to reduce file size",
            default=True
        ),
        decimation_target: int = Input(
            description="Target number of faces for mesh decimation (only if simplify_mesh is True)",
            default=1000000,
            ge=10000,
            le=16777216
        ),
        texture_size: int = Input(
            description="Texture resolution for GLB export",
            default=4096,
            choices=[1024, 2048, 4096]
        ),
        video_fps: int = Input(
            description="Frames per second for output video",
            default=15,
            ge=10,
            le=60
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducibility (leave empty for random)",
            default=None
        ),
    ) -> list[CogPath]:
        """Run image-to-3D generation"""
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            
        # Load input image
        print(f"Loading input image: {image}")
        input_image = Image.open(str(image))
        
        # Run the pipeline
        print(f"Generating 3D mesh at resolution {resolution}...")
        mesh = self.pipeline.run(input_image, resolution=resolution)[0]
        
        # Simplify mesh if requested
        if simplify_mesh:
            print(f"Simplifying mesh (max faces: 16777216)...")
            mesh.simplify(16777216)  # nvdiffrast limit
        
        outputs = []
        
        # Generate video if requested
        if output_format in ["video", "both"]:
            print("Rendering video...")
            video = render_utils.make_pbr_vis_frames(
                render_utils.render_video(mesh, envmap=self.envmap)
            )
            
            video_path = Path(tempfile.mkdtemp()) / "output.mp4"
            imageio.mimsave(str(video_path), video, fps=video_fps)
            outputs.append(CogPath(video_path))
            print(f"Video saved: {video_path}")
        
        # Export GLB if requested
        if output_format in ["glb", "both"]:
            print(f"Exporting GLB (decimation: {decimation_target}, texture: {texture_size})...")
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=True
            )
            
            glb_path = Path(tempfile.mkdtemp()) / "output.glb"
            glb.export(str(glb_path), extension_webp=True)
            outputs.append(CogPath(glb_path))
            print(f"GLB saved: {glb_path}")
        
        return outputs

