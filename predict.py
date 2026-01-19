"""
Cog predictor for TRELLIS2 - Image to 3D Generation
(Updated: runtime-only HF downloads + caching + gated repo support)
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import tempfile
from pathlib import Path
from typing import Optional, List

import cv2
import imageio
import torch
from cog import BasePredictor, Input, Path as CogPath
from PIL import Image

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap

from huggingface_hub import snapshot_download


# -----------------------------
# HF repos / local locations
# -----------------------------
TRELLIS_REPO_ID = "microsoft/TRELLIS.2-4B"
DINO_REPO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # gated
RMBG_REPO_ID = "briaai/RMBG-2.0"                            # gated

MODELS_DIR = os.environ.get("TRELLIS_MODELS_DIR", "/src/models")
TRELLIS_DIR = os.path.join(MODELS_DIR, "TRELLIS.2-4B")
DINO_DIR = os.path.join(MODELS_DIR, "dinov3")
RMBG_DIR = os.path.join(MODELS_DIR, "rmbg")


def _set_hf_cache_env() -> None:
    """
    Use a persistent-ish cache dir (better than /tmp for warm reuse on Replicate).
    """
    hf_home = os.environ.get("HF_HOME", "/root/.cache/huggingface")
    os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # speedup if available

    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def _dir_has_files(p: str) -> bool:
    return os.path.isdir(p) and any(os.scandir(p))


def _ensure_snapshot(repo_id: str, local_dir: str) -> str:
    """
    Download a repo snapshot to local_dir IFF missing/empty.

    - Uses HF_TOKEN / HUGGINGFACE_HUB_TOKEN automatically if present.
    - No interactive login required.
    """
    if _dir_has_files(local_dir):
        return local_dir

    os.makedirs(local_dir, exist_ok=True)

    # snapshot_download will use env token automatically.
    # For gated repos, you must set HF_TOKEN in Replicate secrets.
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    return local_dir


def _enable_offline_if_all_present() -> None:
    """
    If all required repos exist locally, force offline mode to prevent any future
    HF network calls during inference.
    """
    if _dir_has_files(TRELLIS_DIR) and _dir_has_files(DINO_DIR) and _dir_has_files(RMBG_DIR):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory (and download weights lazily at runtime)."""
        _set_hf_cache_env()

        # --- Runtime-only downloads (NO build-time downloads) ---
        # TRELLIS weights (public)
        print("Ensuring TRELLIS.2 weights are available locally...")
        _ensure_snapshot(TRELLIS_REPO_ID, TRELLIS_DIR)

        # These are often pulled indirectly by the pipeline / transformers.
        # If they’re gated and not cached, you’ll get 401s later, so we proactively
        # snapshot them here (still runtime, still cacheable).
        print("Ensuring gated dependency weights are available locally (DINOv3, RMBG)...")
        try:
            _ensure_snapshot(DINO_REPO_ID, DINO_DIR)
        except Exception as e:
            raise RuntimeError(
                "Failed to download DINOv3 (gated). "
                "You must set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in Replicate secrets "
                "and have access approved for the repo."
            ) from e

        try:
            _ensure_snapshot(RMBG_REPO_ID, RMBG_DIR)
        except Exception as e:
            raise RuntimeError(
                "Failed to download RMBG-2.0 (gated). "
                "You must set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in Replicate secrets "
                "and have access approved for the repo."
            ) from e

        # If everything exists, prevent future network calls
        _enable_offline_if_all_present()

        # Optionally expose local dirs for any internal config that checks env vars
        os.environ.setdefault("TRELLIS_DINOV3_PATH", DINO_DIR)
        os.environ.setdefault("TRELLIS_RMBG_PATH", RMBG_DIR)

        # --- Load pipeline from local snapshot path ---
        print("Loading TRELLIS2 pipeline from local snapshot...")
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(TRELLIS_DIR)
        self.pipeline.cuda()

        # --- Load environment map ---
        print("Loading environment map...")
        envmap_path = "assets/hdri/forest.exr"
        self.envmap = EnvMap(
            torch.tensor(
                cv2.cvtColor(cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32,
                device="cuda",
            )
        )

        print("Setup complete!")

    def predict(
        self,
        image: CogPath = Input(description="Input image for 3D generation"),
        output_format: str = Input(
            description="Output format: 'glb' for 3D model, 'video' for visualization, or 'both'",
            default="both",
            choices=["glb", "video", "both"],
        ),
        resolution: int = Input(
            description="Output resolution (512, 1024, or 1536)",
            default=512,
            choices=[512, 1024, 1536],
        ),
        simplify_mesh: bool = Input(
            description="Simplify the mesh to reduce file size",
            default=True,
        ),
        decimation_target: int = Input(
            description="Target number of faces for mesh decimation (only if simplify_mesh is True)",
            default=1000000,
            ge=10000,
            le=16777216,
        ),
        texture_size: int = Input(
            description="Texture resolution for GLB export",
            default=4096,
            choices=[1024, 2048, 4096],
        ),
        video_fps: int = Input(
            description="Frames per second for output video",
            default=15,
            ge=10,
            le=60,
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducibility (leave empty for random)",
            default=None,
        ),
    ) -> List[CogPath]:
        """Run image-to-3D generation"""

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Load input image
        print(f"Loading input image: {image}")
        input_image = Image.open(str(image)).convert("RGB")

        # Run the pipeline
        print(f"Generating 3D mesh at resolution {resolution}...")
        mesh = self.pipeline.run(input_image, resolution=resolution)[0]

        # Simplify mesh if requested
        if simplify_mesh:
            # Keep your original behavior (nvdiffrast limit)
            print("Simplifying mesh (max faces: 16777216)...")
            mesh.simplify(16777216)

        outputs: List[CogPath] = []

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
                verbose=True,
            )

            glb_path = Path(tempfile.mkdtemp()) / "output.glb"
            glb.export(str(glb_path), extension_webp=True)
            outputs.append(CogPath(glb_path))
            print(f"GLB saved: {glb_path}")

        return outputs
