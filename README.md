# SegviGen: Repurposing 3D Generative Model for Part Segmentation

---

## 🖥️ This Fork — Web Application

This fork replaces the original CLI workflow with a **full-stack web application**: a FastAPI backend exposing a job-based API, and a React + TypeScript frontend with interactive 3D viewers.
It was built with the assistance of **[Claude](https://claude.ai)** (Anthropic).

---

## Technical Stack

### Backend
| Component | Technology |
|---|---|
| API server | **FastAPI** + **Uvicorn** |
| Job system | Python `threading` — daemon threads, in-memory job store |
| ML inference | **PyTorch** — all three segmentation models |
| 3D rendering | **Blender** (headless, via subprocess) — renders GLB views for Pixmesh |
| Guidance pipeline | **Gemini API** — VLM describe step + image generation (flat-color segmentation map) |
| 3D I/O | **trimesh**, **Open3D** |
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
FastAPI :7860
    ├── POST /api/upload        → save to temp file, return path
    ├── GET  /api/files         → serve any file by path
    ├── POST /api/jobs/*        → spawn thread → return job_id
    └── GET  /api/jobs/{id}     → poll status / result
            │
            ├── inference.py      (PyTorch segmentation)
            └── guidance_map.py   (Blender render → Gemini → PNG)
```

---

## 🚀 Installation

### Prerequisites
- **System**: Linux
- **GPU**: NVIDIA GPU with at least 12 GB VRAM
- **Python**: 3.10 (3.12 not supported — `bpy` 4.x has no wheels for it)
- **Node.js**: 18+ (for frontend development only)

### 1. Set up the Python environment

Follow the standard SegviGen installation (see below), then install additional dependencies:

```sh
pip install fastapi uvicorn[standard] python-multipart
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
pip install google-genai   # for Pixmesh guidance map generation
```

> **`mathutils` build failure on Python 3.10** — see the patch instructions in the Installation section below.

### 2. Place checkpoints

```
ckpt/
├── interactive_seg.ckpt      ← Interactive Part Segmentation
├── full_seg.ckpt             ← Full Segmentation
└── full_seg_w_2d_map.ckpt    ← Full Segmentation + 2D Guidance
```

Checkpoints are available on [Hugging Face](https://huggingface.co/fenghora/SegviGen).

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
conda activate trellis2   # or your Python 3.10 env
uvicorn server:app --host 0.0.0.0 --port 7860
# → Open http://localhost:7860
```

### Development mode (hot-reload)

Run both processes in separate terminals:

```sh
# Terminal 1 — backend
uvicorn server:app --host 0.0.0.0 --port 7860 --reload

# Terminal 2 — frontend dev server
cd frontend
npm run dev
# → Open http://localhost:5173
```

---

## 🗂️ UI Overview

### Sidebar
- **Input Model** — drag-and-drop or click to upload a GLB / OBJ / PLY file. A live 3D preview appears immediately. The uploaded model path is automatically propagated to every tab.

### Tab: Interactive Part Segmentation
Isolate a single part by specifying one or more 3D voxel click coordinates.

| Field | Description |
|---|---|
| GLB path | Auto-filled from the uploaded model |
| Checkpoint | Path to `interactive_seg.ckpt` |
| Transforms JSON | Camera definition for the rendered view |
| Override rendered image | Optional — skip rendering and supply your own PNG |
| Voxel click points | Space-separated `x y z` triplets (0–511 grid), up to 10 points |

1. Upload your model in the sidebar.
2. Enter the voxel coordinate(s) of the part you want to isolate.
3. Adjust sampler parameters if needed (Steps, CFG, etc.).
4. Click **Run Interactive Segmentation**.
5. Once done, click **Split into Parts** to export each segment as a separate mesh.

### Tab: Full Segmentation
Automatically segments all parts at once, conditioned on a rendered view.

| Field | Description |
|---|---|
| GLB path | Auto-filled from the uploaded model |
| Checkpoint | Path to `full_seg.ckpt` |
| Transforms JSON | Camera definition used to render the conditioning view |
| Override rendered image | Optional — supply your own conditioning PNG |

1. Upload your model.
2. Click **Run Full Segmentation**.
3. Optionally click **Split into Parts**.

### Tab: Full + 2D Map
Combines guidance map generation and 2D-guided segmentation in one place.

#### Step 1 — Generate Guidance Map *(expandable)*
Uses the **Pixmesh** pipeline: renders canonical views with Blender, sends them to a VLM (Gemini) for part description, assigns a Kelly-palette color per part, then uses an image-generation model to flood-fill a flat-color PNG.

| Field | Description |
|---|---|
| GLB path override | Leave empty to use the uploaded model |
| Transforms JSON | Camera for the main conditioning view |
| Gemini API key | Required — `AIza…` key from Google AI Studio |
| Render resolution | 256–1024 px |
| View mode | **Single** (main view only) or **Multi-view grid** (4 canonical views for VLM describe, main view for output) |
| Analyze model | VLM used for the describe step |
| Generate model | Image-generation model used for the flood-fill step |

When generation completes, the result image is shown and automatically pre-fills the guidance path in Step 2.

#### Step 2 — Run Segmentation
| Field | Description |
|---|---|
| GLB path | Auto-filled from the uploaded model |
| Checkpoint | Path to `full_seg_w_2d_map.ckpt` |
| 2D Guidance Map | Auto-filled from Step 1, or browse to upload your own PNG |

1. Expand **Step 1** and generate a guidance map (requires Gemini API key), **or** browse to supply your own flat-color PNG.
2. Click **Run 2D-Guided Segmentation**.
3. Optionally click **Split into Parts**.

### Sampler & Split parameters
All tabs expose two collapsible parameter panels:

**Sampler & Export Parameters**
- Steps, Rescale T, CFG strength, CFG rescale, CFG interval
- Decimation target, texture size, remesh on/off, remesh band, remesh project

**Split Parameters**
- Controls how the segmented mesh is split into individual part files

---

## 🐛 Bug fixes included

- `ColorVisuals → TextureVisuals` conversion before voxelization — GLBs with vertex/flat colors work correctly
- `SimpleMaterial → PBRMaterial` conversion before voxelization — no more `AssertionError` in o_voxel
- `pipeline.json` resolved via `huggingface_hub` instead of a hardcoded relative path
- Multiview guidance fix — grid used only for VLM describe step; segmentation image generated from the single `transforms.json` main view to avoid out-of-distribution interference patterns

---

## 🙏 Credits

- **[Dickoah](https://github.com/Dickoah)** — main contributor for the 2D segmentation pipeline (Pixmesh guidance map generation)
- **[Claude](https://claude.ai)** (Anthropic) — assisted in building the FastAPI backend and React frontend

---

![teaser](assets/teaser.png)

## 🏠 [Project Page](https://fenghora.github.io/SegviGen-Page/) | [Paper](https://arxiv.org/abs/2603.16869) | [Online Demo](https://huggingface.co/spaces/fenghora/SegviGen)

***SegviGen*** is a framework for 3D part segmentation that leverages the rich 3D structural and textural knowledge encoded in large-scale 3D generative models. 
It learns to predict part-indicative colors while reconstructing geometry, and unifies three settings in one architecture: **interactive part segmentation**, **full segmentation**, and **2D segmentation map–guided full segmentation** with arbitrary granularity.


## 🌟 Features
- **Repurposed 3D Generative Priors for Data Efficiency**: By reusing the rich structural and textural knowledge encoded in large-scale native 3D generative models, ***SegviGen*** learns 3D part segmentation with minimal task-specific supervision, requiring only **0.32%** training data.
- **Unified and Flexible Segmentation Settings**: Supports **interactive part segmentation**, **full segmentation**, and **2D segmentation map–guided full segmentation** with arbitrary part granularity under a single architecture.
- **State-of-the-Art Accuracy**: Consistently surpasses P3-SAM, delivering a **40%** gain in IoU@1 for single-click interaction on PartObjaverse-Tiny and PartNeXT, and a **15%** improvement in overall IoU for unguided full segmentation averaged across datasets.


## 🔨 Installation

### Prerequisites
- **System**: Linux
- **GPU**: A NVIDIA GPU with at least 24GB of memory is necessary
- **Python**: 3.10

### Installation Steps
1. Create the environment of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
    ```sh
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
    cd TRELLIS.2
    ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
    ```

2. Install the rest of requirements
    ```sh
    pip install mathutils
    pip install transformers==4.57.6 # https://github.com/microsoft/TRELLIS.2/issues/101
    pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
    sudo apt-get install -y libsm6 libxrender1 libxext6
    pip install --upgrade Pillow
    ```

    > **`mathutils` build failure on Python 3.10** — `mathutils` 5.1.0 uses two Python 3.12+ APIs
    > that are absent in 3.10, causing the wheel build to fail. Apply the following patches to the
    > downloaded source before installing:
    >
    > ```sh
    > pip download mathutils==5.1.0 --no-deps -d /tmp/mathutils_src/
    > cd /tmp && tar -xzf mathutils_src/mathutils-5.1.0.tar.gz && cd mathutils-5.1.0
    >
    > # 1. PyLong_AsInt was added in Python 3.12; replace with (int)PyLong_AsLong
    > sed -i 's/PyLong_AsInt(/(int)PyLong_AsLong(/g' \
    >     src/generic/py_capi_utils.hh src/generic/py_capi_utils.cc
    >
    > # 2. _PyArg_CheckPositional is a static inline in Python < 3.13; guard the re-declaration
    > sed -i 's|^int _PyArg_CheckPositional.*|#if PY_VERSION_HEX >= 0x030d0000\n&\n#endif|' \
    >     src/generic/python_compat.hh
    > sed -i 's|^/\* Removed in Python 3\.13\. \*/|&\n#if PY_VERSION_HEX >= 0x030d0000|' \
    >     src/generic/python_compat.cc
    > printf '\n#endif /* PY_VERSION_HEX >= 0x030d0000 */\n' >> src/generic/python_compat.cc
    >
    > sudo apt-get install -y libeigen3-dev
    > pip install . --no-build-isolation
    > ```


### Pretrained Weights

The checkpoints of **Interactive part-segmentation**, **Full segmentation** and **Full segmentation with 2D guidance** are available on [Hugging Face](https://huggingface.co/fenghora/SegviGen).

## 📒 Usage

- **Interactive part-segmentation**
    ```sh
    python inference_interactive.py \
        --ckpt_path path/to/interactive_seg.ckpt \
        --glb ./data_toolkit/assets/example.glb \
        --input_vxz ./data_toolkit/assets/input.vxz \
        --transforms ./data_toolkit/transforms.json \
        --img ./data_toolkit/assets/img.png \
        --export_glb ./data_toolkit/assets/output.glb \
        --input_vxz_points 388 448 392
    ```

- **Full segmentation**
    ```sh
    python inference_full.py \
        --ckpt_path path/to/full_seg.ckpt \
        --glb ./data_toolkit/assets/example.glb \
        --input_vxz ./data_toolkit/assets/input.vxz \
        --transforms ./data_toolkit/transforms.json \
        --img ./data_toolkit/assets/img.png \
        --export_glb ./data_toolkit/assets/output.glb
    ```

- **Full segmentation with 2D guidance**
    ```sh
    python inference_full.py \
        --ckpt_path path/to/full_seg_w_2d_map.ckpt \
        --glb ./data_toolkit/assets/example.glb \
        --input_vxz ./data_toolkit/assets/input.vxz \
        --img ./data_toolkit/assets/full_seg_w_2d_map/2d_map.png \
        --export_glb ./data_toolkit/assets/output.glb \
        --two_d_map
    ```

## ⚖️ License

This project is licensed under the [MIT License](https://github.com/Nelipot-Lee/SegviGen/blob/main/LICENSE).  
However, please note that the code in **`trellis2`** originates from the [TRELLIS.2](https://github.com/Microsoft/TRELLIS.2) project and remains subject to its original license terms.  
Users must comply with the licensing requirements of [TRELLIS.2](https://github.com/Microsoft/TRELLIS.2) when using or redistributing that portion of the code.

## Citation

```
@article{li2026segvigen,
      title = {SegviGen: Repurposing 3D Generative Model for Part Segmentation}, 
      author = {Lin Li and Haoran Feng and Zehuan Huang and Haohua Chen and Wenbo Nie and Shaohua Hou and Keqing Fan and Pan Hu and Sheng Wang and Buyu Li and Lu Sheng},
      journal = {arXiv preprint arXiv:2603.16869},
      year = {2026}
}
``` 
