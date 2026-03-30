"""segvigen — 3D mesh segmentation using SegviGen models.

Three segmentation modules
--------------------------
- :mod:`segvigen.interactive`   — click-point conditioned segmentation
- :mod:`segvigen.full`          — full segmentation (image-conditioned)
- :mod:`segvigen.full_guided`   — full segmentation conditioned on a 2D guidance map

Supporting modules
------------------
- :mod:`segvigen.presets`  — sampler preset dictionaries

Quick-start
-----------
>>> import segvigen
>>> seg = segvigen.InteractiveSegmenter()
>>> seg.run(glb_path=..., transforms_path=..., points_str=...)
>>> seg = segvigen.FullSegmenter()
>>> seg.run(glb_path=..., transforms_path=...)
>>> seg = segvigen.FullGuidedSegmenter()
>>> seg.run(glb_path=..., guidance_img=...)
"""

# Presets are pure-Python dicts — always importable.
from segvigen.presets import SAMPLER_PRESETS, SPLIT_PRESETS

__all__ = [
    "interactive",
    "full",
    "full_guided",
    "FullSegmenter",
    "FullGuidedSegmenter",
    "InteractiveSegmenter",
    "SAMPLER_PRESETS",
    "SPLIT_PRESETS",
]

# Lazy imports for heavy modules that depend on trellis2 / o_voxel / torch.
# This lets ``import segvigen`` and ``segvigen.SAMPLER_PRESETS`` work even
# without the CUDA prerequisites installed.
_LAZY_IMPORTS = {
    "interactive":          ("segvigen.interactive",  None),
    "full":                 ("segvigen.full",         None),
    "full_guided":          ("segvigen.full_guided",  None),
    "FullSegmenter":        ("segvigen.full",         "FullSegmenter"),
    "FullGuidedSegmenter":  ("segvigen.full_guided",  "FullGuidedSegmenter"),
    "InteractiveSegmenter": ("segvigen.interactive",  "InteractiveSegmenter"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module_path, attr = _LAZY_IMPORTS[name]
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Cannot import segvigen.{name}: {exc}. "
                "The segmenter classes require trellis2, o_voxel, and torch "
                "(CUDA) to be installed. Run install.sh first."
            ) from exc
        return mod if attr is None else getattr(mod, attr)
    raise AttributeError(f"module 'segvigen' has no attribute {name!r}")
