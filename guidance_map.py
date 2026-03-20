"""guidance_map.py — 2D guidance map generators for SegviGen (Tab 3 input).

Each method produces a flat-color segmented PNG where every distinct part of
the 3D model is painted a unique solid color.  Feed the result into the
"Full Segmentation + 2D Guidance Map" tab to steer SegviGen's part boundaries.

Implemented methods
-------------------
- ``run_pixmesh`` : Pixmesh 2D render
    Render one isometric view → VLM describe → assign Kelly palette →
    image-gen model flood-fills each part → save as PNG.
"""

from __future__ import annotations

import base64
import copy
import json
import tempfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

# ── Kelly 22-color palette ─────────────────────────────────────────────────
_KELLY_PALETTE: List[str] = [
    "#FFB300", "#803E75", "#FF6800", "#A6BDD7", "#C10020",
    "#CEA262", "#817066", "#007D34", "#F6768E", "#00538A",
    "#FF7A5C", "#53377A", "#FF8E00", "#B32851", "#F4C800",
    "#7F180D", "#93AA00", "#593315", "#F13A13", "#232C16",
    "#0000FF", "#00FF00",
]


# ── Shared utilities ───────────────────────────────────────────────────────

def _assign_palette(
    description: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Assign Kelly palette colors to every leaf part in the assembly tree.

    Returns ``(updated_description, color_table)`` where ``color_table`` maps
    part name → hex color string.
    """
    updated = copy.deepcopy(description)
    parts_ordered: List[str] = []
    for obj in updated.get("objects", []):
        for group in obj.get("assembly_tree", []):
            for part in group.get("parts", []):
                name = part.get("name", "").strip()
                if name and name not in parts_ordered:
                    parts_ordered.append(name)

    color_table: Dict[str, str] = {
        name: _KELLY_PALETTE[i % len(_KELLY_PALETTE)]
        for i, name in enumerate(parts_ordered)
    }
    # Write assigned colors back into the JSON so prompts are self-consistent.
    for obj in updated.get("objects", []):
        for group in obj.get("assembly_tree", []):
            for part in group.get("parts", []):
                name = part.get("name", "").strip()
                if name in color_table:
                    part["assigned_color_hex"] = color_table[name]

    return updated, color_table


def _img_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ── Pixmesh 2D render — step implementations ──────────────────────────────

def _render_main_view(
    glb_path: str,
    transforms_path: str,
    resolution: int = 512,
) -> Image.Image:
    """Render the first camera in transforms.json via BpyRenderer.

    This reuses the same rendering path used by the Full Segmentation tab
    for its conditioning image (the 3Q isometric view).
    """
    from data_toolkit.bpy_render import render_from_transforms

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        render_path = f.name
    render_from_transforms(
        file_path=glb_path,
        transforms_json_path=transforms_path,
        output_path=render_path,
        resolution=resolution,
    )
    return Image.open(render_path).convert("RGB")


def _gemini_describe(
    image: Image.Image,
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> Dict[str, Any]:
    """Send the rendered view to a Gemini text model; parse and return the JSON assembly tree."""
    system_prompt = (
        "Senior 3D Product Analyst. "
        "Return a SINGLE valid JSON object. No markdown fences, no extra text."
    )
    user_prompt = (
        "Analyze the 3D object in the image and decompose it into its constituent parts.\n\n"
        "Return this exact JSON structure:\n"
        "{\n"
        '  "scene_description": "<max 5 words>",\n'
        '  "language": "en",\n'
        '  "objects": [{\n'
        '    "category": "<Object Name>",\n'
        '    "assembly_tree": [{\n'
        '      "group_name": "<Functional Group>",\n'
        '      "parts": [\n'
        '        {"name": "<Part Name>", "base_color_hex": "<HEX>", "material": "<material>"}\n'
        "      ]\n"
        "    }]\n"
        "  }]\n"
        "}\n\n"
        "Rules:\n"
        "1. List every visually-distinct part as a SEPARATE entry.\n"
        "2. Repeated instances get unique positional names (e.g. 'Leg Front Left').\n"
        "3. Short names only (max 3 words, no color adjectives).\n"
        "4. Do NOT hallucinate parts not visible in the image."
    )
    img_b64 = _img_to_b64(image)
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                {"text": user_prompt},
            ]
        }],
        "generationConfig": {"maxOutputTokens": 8000, "temperature": 0.1},
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    j0, j1 = text.find("{"), text.rfind("}") + 1
    if j0 < 0 or j1 <= j0:
        raise ValueError(f"No JSON object found in VLM response:\n{text[:400]}")
    return json.loads(text[j0:j1])


def _gemini_generate_segmentation(
    image: Image.Image,
    description: Dict[str, Any],
    color_table: Dict[str, str],
    api_key: str,
    model: str = "gemini-2.0-flash-exp-image-generation",
    image_size: Tuple[int, int] = (512, 512),
    bg_color_hex: str = "#ffffff",
) -> Image.Image:
    """Send the rendered view + palette to a Gemini image-gen model.

    Returns the flat-color segmented image.
    """
    W, H = image_size
    n_colors = len(color_table)
    color_table_str = "\n".join(
        f"  {name} → {hex_c}" for name, hex_c in color_table.items()
    )
    json_str = json.dumps(description, indent=2, ensure_ascii=False)

    prompt = (
        "You are a 3D Segmentation Colorist performing a STRICT flat-color recoloring task.\n\n"
        f"## VIEW\nMain isometric view of a 3D object. Image size: {W}×{H} px.\n\n"
        "## JSON ASSEMBLY TREE (for reference only)\n"
        f"```json\n{json_str}\n```\n\n"
        f"## COLOR TABLE — use ONLY these exact hex codes\n"
        f"{color_table_str}\n\n"
        f"Hard limit: at most {n_colors} distinct part colors in the output.\n\n"
        "## RULES\n"
        f"1. Output must be exactly {W}×{H} px — no cropping, no padding.\n"
        f"2. Background ({bg_color_hex}) is NOT a part — leave all background pixels untouched.\n"
        "3. Flood-fill each enclosed region with its single flat hex color.\n"
        "   ONE region = ONE color. No gradients, no outlines, no contour lines in output.\n"
        "4. Object silhouette must be pixel-accurate to the input.\n"
        "5. Do NOT invent new part boundaries or sub-regions.\n"
        f"6. Use ONLY the {n_colors} hex codes above — no other colors are permitted."
    )

    img_b64 = _img_to_b64(image)
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": img_b64}},
            ]
        }],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()

    parts = resp.json()["candidates"][0]["content"]["parts"]
    for part in parts:
        if "inlineData" in part:
            img_data = base64.b64decode(part["inlineData"]["data"])
            img = Image.open(BytesIO(img_data)).convert("RGB")
            if img.size != (W, H):
                img = img.resize((W, H), Image.LANCZOS)
            return img
    raise ValueError(
        "Gemini image generation returned no image. "
        "Check that the model supports image output and the API key is valid.\n"
        f"Raw response: {resp.text[:400]}"
    )


# ── Public entry-point: Pixmesh 2D render ─────────────────────────────────

def run_pixmesh(
    glb_path: str,
    transforms_path: str,
    gemini_api_key: str,
    analyze_model: str = "gemini-2.0-flash",
    generate_model: str = "gemini-2.0-flash-exp-image-generation",
    resolution: int = 512,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[str, Dict[str, Any]]:
    """Pixmesh 2D render pipeline: render → describe → generate → save.

    Returns ``(guidance_map_path, assembly_tree_dict)``.
    """
    if not gemini_api_key or not gemini_api_key.strip():
        raise ValueError("A Gemini API key is required for the Pixmesh method.")

    print("Pixmesh [1/3]: rendering main view …")
    rendered = _render_main_view(glb_path, transforms_path, resolution=resolution)

    print("Pixmesh [2/3]: describing mesh …")
    description = _gemini_describe(rendered, gemini_api_key, model=analyze_model)
    print(f"  → {description.get('scene_description', '?')}")

    updated_desc, color_table = _assign_palette(description)
    print(f"  → {len(color_table)} parts: {', '.join(color_table.keys())}")

    print("Pixmesh [3/3]: generating segmentation image …")
    bg_hex = "#%02x%02x%02x" % bg_color
    seg_image = _gemini_generate_segmentation(
        rendered, updated_desc, color_table,
        api_key=gemini_api_key,
        model=generate_model,
        image_size=(resolution, resolution),
        bg_color_hex=bg_hex,
    )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        out_path = f.name
    seg_image.save(out_path)
    print(f"Pixmesh: done → {out_path}")
    return out_path, updated_desc
