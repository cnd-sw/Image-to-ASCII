"""Quick smoke test for all rendering modes."""
import numpy as np
import cv2
from ascii_converter.core.renderer import Renderer, RenderConfig
from ascii_converter.output.image_out import ImageOutput
from ascii_converter.output.html_out import HtmlOutput
from ascii_converter.core.colorizer import Colorizer

# --------------------------------------------------------------------------
# Build a synthetic test image
# --------------------------------------------------------------------------
img = np.zeros((240, 320, 3), dtype=np.uint8)
for y in range(240):
    img[y, :, 0] = int(y / 240 * 255)
    img[y, :, 2] = int((1 - y / 240) * 200)
cv2.circle(img, (160, 120), 80, (255, 255, 255), -1)
cv2.rectangle(img, (60, 60), (140, 180), (30, 30, 80), -1)
cv2.putText(img, "ASCII", (95, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

# --------------------------------------------------------------------------
# Test each matching mode
# --------------------------------------------------------------------------
modes = ["density", "hybrid", "structure"]
for mode in modes:
    print(f"Testing mode={mode!r} ...", end=" ", flush=True)
    cfg = RenderConfig(
        cols=60,
        chars="standard",
        match_mode=mode,
        color_mode="none",
        dither="none" if mode != "hybrid" else "floyd_steinberg",
    )
    frame = Renderer(cfg).render(img)
    assert frame.rows > 0 and frame.cols == 60, f"Bad frame shape: {frame.rows}x{frame.cols}"
    plain = frame.to_plain_string()
    assert len(plain) > 0
    print(f"OK  ({frame.rows}r x {frame.cols}c)")

# --------------------------------------------------------------------------
# Test colorizer
# --------------------------------------------------------------------------
print("Testing colorizer modes ...", end=" ", flush=True)
for cm in ["none", "ansi16", "ansi256", "truecolor"]:
    cfg = RenderConfig(cols=40, chars="minimal", match_mode="density",
                       color_mode=cm, dither="none")
    frame = Renderer(cfg).render(img)
    ansi = frame.to_ansi_string()
    assert len(ansi) > 0
print("OK")

# --------------------------------------------------------------------------
# Test PNG output
# --------------------------------------------------------------------------
print("Testing PNG output ...", end=" ", flush=True)
cfg = RenderConfig(cols=60, chars="standard", match_mode="density",
                   color_mode="none", dither="none")
frame = Renderer(cfg).render(img)
img_out = ImageOutput(font_size=12)
img_out.save(frame, "/tmp/ascii_test_out.png")
import os
assert os.path.exists("/tmp/ascii_test_out.png")
sz = os.path.getsize("/tmp/ascii_test_out.png")
print(f"OK  ({sz//1024} KB)")

# --------------------------------------------------------------------------
# Test HTML output
# --------------------------------------------------------------------------
print("Testing HTML output ...", end=" ", flush=True)
html_col = Colorizer("html")
frame.fg_codes = html_col.fg_codes(frame.color)
html_out = HtmlOutput(fps=12)
html_out.save([frame], "/tmp/ascii_test_out.html")
assert os.path.exists("/tmp/ascii_test_out.html")
sz = os.path.getsize("/tmp/ascii_test_out.html")
print(f"OK  ({sz//1024} KB)")

# --------------------------------------------------------------------------
# Test all dithering methods
# --------------------------------------------------------------------------
print("Testing dithering ...", end=" ", flush=True)
from ascii_converter.core.dithering import apply_dithering
luma = np.random.rand(20, 20).astype("float32")
for method in ["none", "floyd_steinberg", "stucki", "jarvis", "bayer", "random"]:
    d = apply_dithering(luma.copy(), method=method, levels=10)
    assert d.shape == luma.shape, f"Bad shape for {method}"
    assert 0.0 <= d.min() and d.max() <= 1.0, f"Out of range for {method}"
print("OK")

# --------------------------------------------------------------------------
# Test presets
# --------------------------------------------------------------------------
print("Testing presets ...", end=" ", flush=True)
from ascii_converter.utils.presets import get_preset, list_presets
import dataclasses
valid_fields = {f.name for f in dataclasses.fields(RenderConfig)}
for name in list_presets():
    p = get_preset(name)
    filtered = {k: v for k, v in p.items() if k in valid_fields}
    cfg = RenderConfig(**filtered)
print(f"OK  ({len(list_presets())} presets)")

print()
print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
