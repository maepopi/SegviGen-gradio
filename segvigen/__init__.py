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

from segvigen import interactive, full, full_guided
from segvigen.full import FullSegmenter
from segvigen.full_guided import FullGuidedSegmenter
from segvigen.interactive import InteractiveSegmenter
from segvigen.presets import SAMPLER_PRESETS

__all__ = [
    "interactive",
    "full",
    "full_guided",
    "FullSegmenter",
    "FullGuidedSegmenter",
    "InteractiveSegmenter",
    "SAMPLER_PRESETS",
]
