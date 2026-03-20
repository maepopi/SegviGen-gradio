"""server.py — FastAPI server for SegviGen.

Run with:
    uvicorn server:app --host 0.0.0.0 --port 7860 --reload
"""

import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import inference as inf
import guidance_map as gmap

# ─── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="SegviGen")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ─── Job store ─────────────────────────────────────────────────────────────────
# Simple in-memory job dict — good enough for local single-user tool.

_jobs: Dict[str, Dict[str, Any]] = {}


def _run_job(job_id: str, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        _jobs[job_id] = {"status": "done", "result": result, "error": None}
    except Exception as exc:
        _jobs[job_id] = {"status": "error", "result": None, "error": str(exc)}


def _start_job(fn, *args, **kwargs) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "result": None, "error": None}
    t = threading.Thread(target=_run_job, args=(job_id, fn, *args), kwargs=kwargs, daemon=True)
    t.start()
    return {"job_id": job_id}


# ─── Root ───────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ─── File upload ────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile):
    suffix = Path(file.filename).suffix if file.filename else ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        shutil.copyfileobj(file.file, f)
        return {"path": f.name}


# ─── File download ──────────────────────────────────────────────────────────────

@app.get("/api/files")
def serve_file(path: str):
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


# ─── Job status ─────────────────────────────────────────────────────────────────

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


# ─── Presets ────────────────────────────────────────────────────────────────────

@app.get("/api/presets/sampler")
def sampler_presets():
    return inf.SAMPLER_PRESETS


@app.get("/api/presets/split")
def split_presets():
    return inf.SPLIT_PRESETS


# ─── Inference endpoints ────────────────────────────────────────────────────────

class InteractiveParams(BaseModel):
    glb_path: str
    ckpt_path: str
    transforms_path: str
    rendered_img: Optional[str] = None
    points_str: str = "388 448 392"
    steps: int = 25
    rescale_t: float = 1.0
    guidance_strength: float = 7.5
    guidance_rescale: float = 0.0
    guidance_interval_start: float = 0.0
    guidance_interval_end: float = 1.0
    decimation_target: int = 100_000
    texture_size: int = 1024
    remesh: bool = True
    remesh_band: int = 1
    remesh_project: int = 0


class FullParams(BaseModel):
    glb_path: str
    ckpt_path: str
    transforms_path: str
    rendered_img: Optional[str] = None
    steps: int = 25
    rescale_t: float = 1.0
    guidance_strength: float = 7.5
    guidance_rescale: float = 0.0
    guidance_interval_start: float = 0.0
    guidance_interval_end: float = 1.0
    decimation_target: int = 100_000
    texture_size: int = 1024
    remesh: bool = True
    remesh_band: int = 1
    remesh_project: int = 0


class Full2DParams(BaseModel):
    glb_path: str
    ckpt_path: str
    guidance_img: str
    steps: int = 25
    rescale_t: float = 1.0
    guidance_strength: float = 7.5
    guidance_rescale: float = 0.0
    guidance_interval_start: float = 0.0
    guidance_interval_end: float = 1.0
    decimation_target: int = 100_000
    texture_size: int = 1024
    remesh: bool = True
    remesh_band: int = 1
    remesh_project: int = 0


class SplitParams(BaseModel):
    seg_glb_path: str
    color_quant_step: int = 16
    palette_sample_pixels: int = 2_000_000
    palette_min_pixels: int = 500
    palette_max_colors: int = 256
    palette_merge_dist: int = 32
    samples_per_face: int = 4
    flip_v: bool = True
    uv_wrap_repeat: bool = True
    transition_conf_thresh: float = 1.0
    transition_prop_iters: int = 6
    transition_neighbor_min: int = 1
    small_component_action: str = "reassign"
    small_component_min_faces: int = 50
    postprocess_iters: int = 3
    min_faces_per_part: int = 1
    bake_transforms: bool = True


class GuidanceParams(BaseModel):
    glb_path: str
    transforms_path: str
    gemini_api_key: str
    analyze_model: str = "gemini-2.5-flash"
    generate_model: str = "gemini-3-pro-image-preview"
    resolution: int = 512
    mode: str = "single"
    grid_views: List[str] = ["front", "back", "left", "right"]
    grid_cols: int = 2


@app.post("/api/jobs/interactive")
def start_interactive(params: InteractiveParams):
    return _start_job(inf.run_interactive, **params.model_dump())


@app.post("/api/jobs/full")
def start_full(params: FullParams):
    return _start_job(inf.run_full, **params.model_dump())


@app.post("/api/jobs/full_2d")
def start_full_2d(params: Full2DParams):
    return _start_job(inf.run_full_2d, **params.model_dump())


@app.post("/api/jobs/split")
def start_split(params: SplitParams):
    return _start_job(inf.run_split, **params.model_dump())


@app.post("/api/jobs/guidance")
def start_guidance(params: GuidanceParams):
    def _run():
        out_path, description = gmap.run_pixmesh(
            glb_path=params.glb_path,
            transforms_path=params.transforms_path,
            gemini_api_key=params.gemini_api_key,
            analyze_model=params.analyze_model,
            generate_model=params.generate_model,
            resolution=params.resolution,
            mode=params.mode,
            grid_views=tuple(params.grid_views),
            grid_cols=params.grid_cols,
        )
        return {"image_path": out_path, "description": description}
    return _start_job(_run)


# ─── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
