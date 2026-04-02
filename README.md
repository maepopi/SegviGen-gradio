# SegviGen: Repurposing 3D Generative Model for Part Segmentation

---

## 🖥️ This Fork — Web Application

This fork replaces the original CLI workflow with a **full-stack web application**: a FastAPI backend exposing a job-based API, and a React + TypeScript frontend with interactive 3D viewers.

---

## Technical Stack

### Backend
| Component | Technology |
|---|---|
| API server | **FastAPI** + **Uvicorn** |
| Job system | Python `threading` — daemon threads, in-memory job store |
| ML inference | **segvigen** package — lazy-loading segmenter classes with per-instance locking and automatic VRAM management |
| 3D rendering | **Blender** (headless, via `bpy`) — renders GLB conditioning views |
| Guidance pipeline | **Gemini API** — VLM describe step + image generation (flat-color segmentation map) |
| Background removal | **rembg** (`isnet-general-use`) — optional, injected into segmenters via `remove_bg_fn` callback |
| 3D I/O | **trimesh**, **o_voxel** |
| Data validation | **Pydantic** v2 |

### Frontend
| Component | Technology |
|---|---|
| Framework | **React 18** + **TypeScript** |
| Build tool | **Vite** |
| Styling | **Tailwind CSS** v3 (custom dark theme) |
| 3D viewer | **`<model-viewer>`** (Google web component) |
| Icons | **Lucide React** |
| HTTP | Native `fetch` |

### Architecture
```
Browser (Vite dev :5173 / or FastAPI static)
    │  /api/*
    ▼
FastAPI :7860  (server.py)
    ├── POST /api/upload        → save to temp file, return path
    ├── GET  /api/files         → serve any file by path
    ├── POST /api/jobs/*        → spawn thread → return job_id
    └── GET  /api/jobs/{id}     → poll status / result
            │
            ├── segvigen/                 (core ML package)
            │   ├── FullSegmenter         → full.py
            │   ├── FullGuidedSegmenter   → full_guided.py
            │   ├── InteractiveSegmenter  → interactive.py
            │   ├── _shared.py            (model loading, voxel I/O, GLB export)
            │   └── _samplers.py          (flow-matching samplers, DiT wrappers)
            │
            └── util.py
                ├── remove_bg()           (rembg background removal)
                ├── split_glb_by_texture_palette_rgb()
                └── generate_guidance_map()  (Blender render → Gemini → PNG)
```

#### `segvigen` module design
Each segmenter class follows the same lifecycle:
1. **Lazy loading** — models are downloaded and loaded on first `run()` call
2. **Thread-safe** — a per-instance `threading.Lock` serializes concurrent requests
3. **VRAM management** — sub-models are moved to CUDA one at a time during inference, then offloaded to CPU; `clear_vram()` runs in a `finally` block
4. **Temp file cleanup** — intermediate `.vxz` and `.png` files are deleted after each run; output `.glb` is left for the caller
5. **Weight caching** — heavy model weights are cached globally in `_shared.py`, so multiple segmenter instances sharing the same checkpoint reuse the same weights

---

## 🚀 Installation

### Prerequisites
- **System**: Linux
- **GPU**: NVIDIA GPU with at least 12 GB VRAM
- **Python**: 3.11 (managed by the install script — do not use 3.12+)
- **CUDA**: 12.x
- **Conda**: miniconda or anaconda
- **Node.js**: 18+ (for frontend development only)

### 1. Run the install script

The script handles everything: creates a `segvigen` conda env (Python 3.11), builds all TRELLIS.2 CUDA extensions, installs all dependencies, and downloads missing checkpoints automatically.

```sh
bash install.sh
```

> **What the script does, step by step:**
> 1. Clones TRELLIS.2 (skips if already present)
> 2. Creates conda env `segvigen` with Python 3.11, installs PyTorch cu128
> 3. Builds TRELLIS.2 CUDA extensions (`o_voxel`, `cumesh`, `flex_gemm`, `nvdiffrast`, `nvdiffrec`) — takes 30–60 min
> 4. Installs SegviGen Python deps (including patched `mathutils`, `bpy 4.0.0`, `gradio 6.0.1`, `Pillow 10.x`)
> 5. Installs system libs (`libsm6`, `libopenexr-dev`, etc.)
> 6. Downloads any missing checkpoints from [HuggingFace](https://huggingface.co/fenghora/SegviGen) into `ckpt/`

> **Note on `mathutils`:** The install script automatically patches `mathutils 5.1.0` source to compile on Python 3.11 (fixes `PyLong_AsInt` and `_PyArg_CheckPositional` compatibility issues).

> **Note on Pillow:** `gradio 6.0.1` requires `Pillow < 11` (`HAVE_WEBPANIM` was removed in Pillow 11). The script pins `Pillow>=10,<11` and removes `pillow-simd` if installed by TRELLIS.2.

### Alternative: pip install (library only)

If you already have the TRELLIS.2 CUDA prerequisites (`trellis2`, `o_voxel`, `torch`) installed in your environment, you can install just the `segvigen` Python package:

```sh
pip install git+https://github.com/maepopi/SegviGen-app
```

> **Note:** This installs only the `segvigen` package and its PyPI dependencies. It does **not** build the CUDA extensions (`trellis2`, `o_voxel`, etc.) — those must already be present in your environment. If they are missing, `import segvigen` will still work for lightweight usage (e.g. accessing presets), but the segmenter classes will raise a clear `ImportError` when accessed.

### 2. Place checkpoints

Checkpoints are downloaded automatically by `install.sh`. If you need to download them manually:

```sh
conda activate segvigen
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
import shutil, os
ckpt_dir = 'ckpt'
os.makedirs(ckpt_dir, exist_ok=True)
for f in ['interactive_seg.ckpt', 'full_seg.ckpt', 'full_seg_w_2d_map.ckpt']:
    shutil.copy(hf_hub_download('fenghora/SegviGen', f), os.path.join(ckpt_dir, f))
"
```

Expected layout:

```
ckpt/
├── interactive_seg.ckpt      ← Interactive Part Segmentation
├── full_seg.ckpt             ← Full Segmentation
└── full_seg_w_2d_map.ckpt    ← Full Segmentation + 2D Guidance
```

### 3. (Optional) Build the frontend

The `static/` directory already contains a pre-built frontend. To rebuild from source:

```sh
cd frontend
npm install
npm run build   # outputs to ../static/
```

---

## 📖 Usage

### Starting the server

```sh
conda activate segvigen   # Python 3.11 env created by install.sh
uvicorn server:app --host 0.0.0.0 --port 7860
# → Open http://localhost:7860
```

### Development mode (hot-reload)

Run both processes in separate terminals:

```sh
# Terminal 1 — backend
conda activate segvigen
uvicorn server:app --host 0.0.0.0 --port 7860 --reload

# Terminal 2 — frontend dev server
cd frontend
npm run dev
# → Open http://localhost:5173
```

---

##  Bug fixes included

- `ColorVisuals → TextureVisuals` conversion before voxelization — GLBs with vertex/flat colors work correctly
- `SimpleMaterial → PBRMaterial` conversion before voxelization — no more `AssertionError` in o_voxel
- `pipeline.json` resolved via `huggingface_hub` instead of a hardcoded relative path
- Multiview guidance fix — grid used only for VLM describe step; segmentation image generated from the single `transforms.json` main view to avoid out-of-distribution interference patterns
- Proper RGBA handling in `preprocess_image` — images with pre-applied alpha (e.g., user-supplied background-removed images) are used directly instead of being re-processed through background removal

---

## 🙏 Credits

- **[SegviGen](https://github.com/Nelipot-Lee/SegviGen)** — original research and codebase by Lin Li, Haoran Feng, Zehuan Huang, Haohua Chen, Wenbo Nie, Shaohua Hou, Keqing Fan, Pan Hu, Sheng Wang, Buyu Li, and Lu Sheng
- **[Dickoah](https://github.com/Dickoah)** — my dear friend, and main contributor for the 2D segmentation pipeline (guidance map generation)

---

![teaser](assets/teaser.png)

## 🏠 [Project Page](https://fenghora.github.io/SegviGen-Page/) | [Paper](https://arxiv.org/abs/2603.16869) | [Online Demo](https://huggingface.co/spaces/fenghora/SegviGen)

***SegviGen*** is a framework for 3D part segmentation that leverages the rich 3D structural and textural knowledge encoded in large-scale 3D generative models. 
It learns to predict part-indicative colors while reconstructing geometry, and unifies three settings in one architecture: **interactive part segmentation**, **full segmentation**, and **2D segmentation map–guided full segmentation** with arbitrary granularity.

## Citation

```
@article{li2026segvigen,
      title = {SegviGen: Repurposing 3D Generative Model for Part Segmentation}, 
      author = {Lin Li and Haoran Feng and Zehuan Huang and Haohua Chen and Wenbo Nie and Shaohua Hou and Keqing Fan and Pan Hu and Sheng Wang and Buyu Li and Lu Sheng},
      journal = {arXiv preprint arXiv:2603.16869},
      year = {2026}
}
``` 
