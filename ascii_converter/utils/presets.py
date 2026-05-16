"""
presets.py
----------
Named style presets that configure the full conversion pipeline.
Each preset is a dict of overrides applied on top of the default config.
"""

from __future__ import annotations
from typing import Dict, Any

PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "high_detail": {
        "cols": 200, "edge_boost": 0.4, "edge_method": "sobel",
        "contrast": 1.2, "normalize": True, "dither": "floyd_steinberg",
        "match_mode": "hybrid", "chars": "unicode_block", "color_mode": "truecolor",
    },
    "retro": {
        "cols": 80, "chars": "standard", "color_mode": "ansi16",
        "dither": "none", "contrast": 1.3, "gamma": 0.9, "edge_boost": 0.2,
    },
    "artistic": {
        "cols": 120, "edge_boost": 0.6, "edge_method": "canny",
        "dither": "bayer", "chars": "extended", "contrast": 1.4,
        "saturation": 1.3, "color_mode": "truecolor", "match_mode": "hybrid",
    },
    "minimal": {
        "cols": 80, "chars": "minimal", "dither": "none", "edge_boost": 0.0,
        "contrast": 1.0, "normalize": False, "color_mode": "none", "match_mode": "density",
    },
    "anime": {
        "cols": 160, "chars": "braille", "saturation": 1.5, "edge_boost": 0.5,
        "edge_method": "canny", "canny_low": 30, "canny_high": 100,
        "dither": "none", "contrast": 1.2, "color_mode": "truecolor", "match_mode": "hybrid",
    },
    "matrix": {
        "cols": 100, "chars": "digits", "color_mode": "ansi16",
        "dither": "random", "contrast": 1.5, "gamma": 0.8,
    },
    "braille": {
        "cols": 160, "chars": "braille", "color_mode": "truecolor",
        "dither": "floyd_steinberg", "match_mode": "density", "edge_boost": 0.2,
    },
    "block": {
        "cols": 120, "chars": "block", "color_mode": "truecolor",
        "dither": "bayer", "match_mode": "density",
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}. Available: {available}")
    return dict(PRESETS[name])


def list_presets() -> list:
    return sorted(PRESETS.keys())
