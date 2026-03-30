"""segvigen.interactive — Interactive (click-point) 3D segmentation.

Usage
-----
>>> from segvigen.interactive import InteractiveSegmenter
>>> seg = InteractiveSegmenter(ckpt_path="ckpt/interactive_seg.ckpt")
>>> out_glb = seg.run(
...     glb_path="model.glb",
...     transforms_path="data_toolkit/transforms.json",
...     points_str="388 448 392",
... )
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from huggingface_hub import hf_hub_download

import trellis2.modules.sparse as sp

from segvigen._shared import (
    _to_cuda, _offload,
    process_glb_to_vxz, vxz_to_latent_slat,
    preprocess_image, get_cond, slat_to_glb,
    load_base_models, load_seg_model, build_sampler_params,
)
from segvigen._samplers import SamplerInteractive

_HF_REPO = "fenghora/SegviGen"
_DEFAULT_CKPT = "ckpt/interactive_seg.ckpt"


class InteractiveSegmenter:
    """Interactive (click-point) 3D segmentation.

    Models are loaded lazily on the first call to :meth:`run`.

    Parameters
    ----------
    ckpt_path:
        Path to ``interactive_seg.ckpt``.
    """

    def __init__(self, ckpt_path: str = _DEFAULT_CKPT) -> None:
        self.ckpt_path = ckpt_path
        self._base = None
        self._gen3dseg = None
        self._sampler = None
        self._lock = threading.Lock()

    def _ensure_checkpoint(self) -> None:
        """Download the checkpoint from HuggingFace if it is missing."""
        if os.path.isfile(self.ckpt_path):
            return
        print(f"[InteractiveSegmenter] Checkpoint not found at {self.ckpt_path!r} — downloading …")
        local_dir = str(Path(self.ckpt_path).parent)
        hf_hub_download(repo_id=_HF_REPO, filename=Path(self.ckpt_path).name,
                        local_dir=local_dir)
        print(f"[InteractiveSegmenter] Downloaded {self.ckpt_path}")

    def load(self) -> None:
        """Load base models and the segmentation model into CPU memory."""
        if self._base is not None:
            return
        self._ensure_checkpoint()
        self._base = load_base_models()
        self._gen3dseg = load_seg_model(self.ckpt_path, 'interactive')
        self._sampler = SamplerInteractive()

    def clear_vram(self) -> None:
        """Move all model weights to CPU and flush the CUDA cache."""
        if self._base is not None:
            for m in self._base.values():
                if isinstance(m, torch.nn.Module):
                    _offload(m)
        if self._gen3dseg is not None:
            _offload(self._gen3dseg)
        torch.cuda.empty_cache()

    def run(
        self,
        glb_path: str,
        transforms_path: str,
        rendered_img: Optional[str] = None,
        points_str: str = "388 448 392",
        remove_bg_fn=None,
        steps: int = 25,
        rescale_t: float = 1.0,
        guidance_strength: float = 7.5,
        guidance_rescale: float = 0.0,
        guidance_interval_start: float = 0.0,
        guidance_interval_end: float = 1.0,
        decimation_target: int = 100_000,
        texture_size: int = 1024,
        remesh: bool = True,
        remesh_band: int = 1,
        remesh_project: int = 0,
    ) -> str:
        """Run interactive segmentation on a GLB mesh.

        Parameters
        ----------
        glb_path:
            Input GLB file path.
        transforms_path:
            Path to ``transforms.json`` used for rendering the conditioning view.
        rendered_img:
            Optional pre-rendered PNG to use instead of the auto-rendered view.
        points_str:
            Space-separated ``x y z`` voxel coordinates (multiples of 3).
            Example: ``"388 448 392  200 300 250"``.
        steps, rescale_t, guidance_strength, guidance_rescale,
        guidance_interval_start, guidance_interval_end:
            Diffusion sampler parameters.
        decimation_target, texture_size, remesh, remesh_band, remesh_project:
            GLB export parameters.

        Returns
        -------
        str
            Absolute path to the segmented output GLB.
        """
        try:
            from data_toolkit.bpy_render import render_from_transforms
        except ImportError as exc:
            raise ImportError(
                "data_toolkit.bpy_render is required for rendering the conditioning view. "
                "Make sure the data_toolkit module is on your Python path."
            ) from exc

        with self._lock:
            self.load()
            base = self._base
            gen3dseg = self._gen3dseg
            sampler = self._sampler

            vxz_path = img_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.vxz', delete=False) as f:
                    vxz_path = f.name
                with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
                    out_path = f.name
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    img_path = f.name

                print("GLB → VXZ …")
                process_glb_to_vxz(glb_path, vxz_path)
                shape_slat, meshes, subs, tex_slat = vxz_to_latent_slat(
                    base['shape_encoder'], base['shape_decoder'], base['tex_encoder'], vxz_path)

                print("Rendering conditioning image …")
                render_from_transforms(glb_path, transforms_path, img_path)
                if rendered_img is not None:
                    img_path = rendered_img
                image = Image.open(img_path)
                image = preprocess_image(image, remove_bg_fn=remove_bg_fn)
                _to_cuda(base['image_cond_model'])
                cond = get_cond(base['image_cond_model'], [image])
                _offload(base['image_cond_model'])

                flat = [int(v) for v in points_str.split()]
                if len(flat) % 3 != 0:
                    raise ValueError("points_str must contain coordinates in multiples of 3 (x y z per point).")
                input_vxz_points_list = [flat[i:i + 3] for i in range(0, len(flat), 3)]

                print("Encoding click points …")
                vxz_points_coords = torch.tensor(input_vxz_points_list, dtype=torch.int32).cuda()
                vxz_points_coords = torch.cat(
                    [torch.zeros((vxz_points_coords.shape[0], 1), dtype=torch.int32).cuda(), vxz_points_coords], dim=1)
                _to_cuda(base['tex_encoder'])
                input_points_coords = base['tex_encoder'](
                    sp.SparseTensor(torch.zeros((vxz_points_coords.shape[0], 6), dtype=torch.float32).cuda(),
                                    vxz_points_coords)).coords
                _offload(base['tex_encoder'])
                input_points_coords = torch.unique(input_points_coords, dim=0)
                point_num = input_points_coords.shape[0]
                if point_num >= 10:
                    input_points_coords = input_points_coords[:10]
                    point_labels = torch.tensor([[1]] * 10, dtype=torch.int32).cuda()
                else:
                    input_points_coords = torch.cat(
                        [input_points_coords,
                         torch.zeros((10 - point_num, 4), dtype=torch.int32).cuda()], dim=0)
                    point_labels = torch.tensor(
                        [[1]] * point_num + [[0]] * (10 - point_num), dtype=torch.int32).cuda()
                input_points = {
                    'point_slats': sp.SparseTensor(input_points_coords, input_points_coords),
                    'point_labels': point_labels,
                }

                sampler_params = build_sampler_params(
                    base['pipeline_args'], steps, rescale_t, guidance_strength,
                    guidance_rescale, guidance_interval_start, guidance_interval_end)

                print("Sampling …")
                pa = base['pipeline_args']
                device = shape_slat.feats.device
                shape_std = torch.tensor(pa['shape_slat_normalization']['std'])[None].to(device)
                shape_mean = torch.tensor(pa['shape_slat_normalization']['mean'])[None].to(device)
                tex_std = torch.tensor(pa['tex_slat_normalization']['std'])[None].to(device)
                tex_mean = torch.tensor(pa['tex_slat_normalization']['mean'])[None].to(device)
                shape_slat_n = (shape_slat - shape_mean) / shape_std
                tex_slat_n = (tex_slat - tex_mean) / tex_std
                coords_len_list = [shape_slat_n.coords.shape[0]]
                noise = sp.SparseTensor(torch.randn_like(tex_slat_n.feats), shape_slat_n.coords)
                _to_cuda(gen3dseg)
                output_tex_slat = sampler.sample(gen3dseg, noise, tex_slat_n, shape_slat_n,
                                                  input_points, coords_len_list, cond, sampler_params)
                _offload(gen3dseg)
                output_tex_slat = output_tex_slat * tex_std + tex_mean

                _to_cuda(base['tex_decoder'])
                with torch.no_grad():
                    tex_voxels = base['tex_decoder'](output_tex_slat, guide_subs=subs) * 0.5 + 0.5
                _offload(base['tex_decoder'])

                print("Exporting GLB …")
                glb = slat_to_glb(meshes, tex_voxels,
                                   decimation_target=int(decimation_target),
                                   texture_size=int(texture_size),
                                   remesh=remesh,
                                   remesh_band=remesh_band,
                                   remesh_project=remesh_project)
                glb.export(out_path)
                return out_path
            finally:
                self.clear_vram()
                for _p in (vxz_path, img_path):
                    if _p is not None:
                        try:
                            os.unlink(_p)
                        except OSError:
                            pass
