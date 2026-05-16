# ASCII Art Converter

> **Production-quality, local-only ASCII / ANSI art converter for images and videos.**
> Surpasses existing tools (jp2a, chafa, ascii-image-converter, ascii_magic) in detail, colour fidelity, performance, and output options.

---

## ✨ Feature Highlights

| Feature | Details |
|---------|---------|
| **Character matching** | Density · HoG-structure · Hybrid (density+HoG) · ML (kNN/RF) |
| **Character sets** | ASCII · Extended · Unicode block/box · Braille · Half-block |
| **Colour modes** | Truecolor ANSI · ANSI-256 · ANSI-16 · HTML/CSS · None |
| **Dithering** | Floyd-Steinberg · Stucki · Jarvis · Bayer 8×8 · Random |
| **Edge enhancement** | Canny · Sobel · Laplacian with adjustable boost |
| **Preprocessing** | Gamma · Contrast · Brightness · Saturation · Hue · Bilateral filter |
| **Video** | Frame-by-frame, multiprocessing, temporal smoothing |
| **Output** | Terminal · PNG · MP4 (FFmpeg) · GIF · HTML+JS animation |
| **UI** | Rich CLI (Click) + Gradio web UI |
| **Performance** | NumPy/OpenCV vectorised · multiprocessing · disk-cached character library |

---

## 📦 Installation

### 1. Clone / download the project

```bash
git clone <repo-url> "Image to ASCII"
cd "Image to ASCII"
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Optional extras:**
> - **GPU support** (CuPy): `pip install cupy-cuda12x`
> - **Better font rendering**: `pip install freetype-py` *(already in requirements)*
> - **ML matching**: `pip install scikit-learn` *(already in requirements)*
> - **Web UI**: `pip install gradio` *(already in requirements)*

### 4. Verify FFmpeg (needed for MP4 export)

```bash
ffmpeg -version
```

If missing: `brew install ffmpeg` (macOS) · `sudo apt install ffmpeg` (Linux)

---

## 🚀 Quick Start

### Image → Terminal (live preview)

```bash
python main.py image photo.jpg
```

### Image → High-quality terminal (truecolor + hybrid matching)

```bash
python main.py image photo.jpg --preset high_detail --cols 180 --color truecolor
```

### Image → PNG file

```bash
python main.py image photo.jpg --out output.png --cols 200 --preset high_detail
```

### Image → HTML (colour, zoomable in browser)

```bash
python main.py image photo.jpg --out output.html --color html --cols 160
```

### Image → Plain text file

```bash
python main.py image photo.jpg --out output.txt --cols 120
```

---

## 🎬 Video Examples

### Video → Terminal live view

```bash
python main.py video clip.mp4 --fps 15 --cols 120
```

### Video → MP4 with audio

```bash
python main.py video clip.mp4 --out ascii_clip.mp4 --fps 24 --preset artistic
```

### Video → Animated GIF

```bash
python main.py video clip.mp4 --out ascii.gif --fps 12 --cols 100
```

### Video → HTML animation (self-contained, shareable)

```bash
python main.py video clip.mp4 --out ascii.html --fps 15 --preset braille
```

### Webcam live ASCII art

```bash
python main.py webcam --cols 100 --preset anime
```

---

## 🌐 Web UI (Gradio)

```bash
python main.py ui
# Opens at http://localhost:7860
```

Or with a public share link (tunnels through Gradio servers):

```bash
python main.py ui --share
```

---

## ⚙️ All CLI Options

### Shared rendering options (available on `image`, `video`, `webcam`)

| Option | Default | Description |
|--------|---------|-------------|
| `--preset` | `default` | Style preset name |
| `--cols N` | `120` | Character columns |
| `--rows N` | auto | Character rows (auto = aspect-correct) |
| `--mode` | `hybrid` | Matching: `density` `structure` `hybrid` `ml` |
| `--chars` | `standard` | Character set |
| `--color` | `truecolor` | Colour mode |
| `--dither` | `floyd_steinberg` | Dithering algorithm |
| `--edge-boost F` | `0.3` | Edge enhancement strength 0–1 |
| `--contrast F` | `1.0` | Contrast multiplier |
| `--brightness F` | `0.0` | Brightness offset −1…1 |
| `--gamma F` | `1.0` | Gamma correction |
| `--saturation F` | `1.0` | Saturation multiplier |
| `--invert` | off | Invert luminance |
| `--bilateral` | off | Bilateral smoothing filter |
| `--font-path PATH` | auto | TTF font for PNG/MP4 rendering |
| `--font-size N` | `14` | Font size in points |
| `--verbose` / `-v` | off | Enable debug logging |

### `video`-specific options

| Option | Default | Description |
|--------|---------|-------------|
| `--fps F` | source | Target output FPS |
| `--max-frames N` | all | Limit number of frames |
| `--workers N` | auto | Parallel worker processes |
| `--temporal F` | `0.0` | Temporal blend factor (reduces flicker) |
| `--audio/--no-audio` | on | Include audio in MP4 |

---

## 🎨 Style Presets

```bash
python main.py presets   # list all presets
```

| Preset | Description |
|--------|-------------|
| `default` | Balanced defaults — good starting point |
| `high_detail` | 200 cols, Sobel edges, hybrid — best for complex scenes |
| `retro` | 80 cols, ASCII, 16-colour ANSI — classic green terminal |
| `artistic` | 120 cols, Canny, Bayer dither — graphic-novel vibe |
| `minimal` | 80 cols, 10 chars, no colour — ultra-clean |
| `anime` | 160 cols, braille, vivid saturation — flat-colour anime |
| `matrix` | 100 cols, digits, ANSI-16 — Matrix rain |
| `braille` | 160 cols, full braille charset — maximum density |
| `block` | 120 cols, half-block Unicode — pixel-art fidelity |

---

## 🏗️ Project Architecture

```
Image to ASCII/
├── main.py                          # CLI entry point (Click)
├── requirements.txt
├── config.yaml                      # Default configuration
├── web_ui/
│   ├── __init__.py
│   └── app.py                       # Gradio web UI
└── ascii_converter/
    ├── core/
    │   ├── preprocessor.py          # Resize, luminance, edges, tone mapping
    │   ├── char_library.py          # Render & cache character glyph patches
    │   ├── char_matcher.py          # Density / HoG / Hybrid / ML matching
    │   ├── colorizer.py             # ANSI truecolor / 256 / 16 / HTML codes
    │   ├── dithering.py             # FS / Stucki / Jarvis / Bayer / random
    │   └── renderer.py              # Main pipeline → AsciiFrame
    ├── video/
    │   ├── processor.py             # Frame extraction, multiprocessing, webcam
    │   └── exporter.py              # FFmpeg MP4 + GIF export
    ├── output/
    │   ├── terminal.py              # ANSI live terminal display
    │   ├── image_out.py             # PNG rendering via Pillow
    │   ├── html_out.py              # Self-contained HTML+JS animation
    │   └── gif_out.py               # Animated GIF via Pillow
    ├── ml/
    │   └── char_model.py            # kNN / Random Forest classifier
    └── utils/
        ├── font_metrics.py          # Real font aspect ratio (freetype-py)
        └── presets.py               # Named style presets
```

---

## ⚡ Performance Tips

1. **Use `--mode density`** for maximum speed (1–2 ms/frame on 120-col output).
2. **Use `--mode hybrid`** for best quality/speed balance (~5–15 ms/frame).
3. **Increase `--workers`** for video: `--workers 8` on an 8-core CPU gives ~5× speedup.
4. **Lower `--cols`** for real-time video: 80 cols is typically real-time on any modern CPU.
5. **The character library is cached** after first build — subsequent runs are instant.
6. **GPU (CuPy)**: Replace NumPy operations with CuPy for 10–50× speedup on CUDA GPUs (manual integration step — swap `np` → `cp` in `preprocessor.py` and `char_matcher.py`).
7. **Use `--dither none`** for fastest processing.
8. **`--bilateral`** adds quality for noisy/compressed images but costs ~3× preprocessing time.

---

## 🔬 How the Hybrid Matching Works

1. **Preprocess**: resize image to character grid, compute luminance + edge map.
2. **Dither**: apply selected error-diffusion or ordered dithering to spread quantisation error.
3. **Density pre-filter**: for each cell, use binary search on the sorted density array to find the `topk=8` nearest characters by brightness.
4. **HoG re-rank**: compute a gradient-orientation histogram for the cell, then select the candidate with highest cosine similarity to the character's pre-computed HoG descriptor.
5. **Colour**: map the average pixel colour of each cell to the selected colour mode.

This hybrid approach avoids the `O(N)` full search while still considering structure — giving near-ML quality at density-matching speed.

---

## 🤝 Contributing

- Fork, make changes, open a PR.
- Run `python main.py image <test.jpg> --verbose` to verify your changes.
- Keep all operations vectorised (no Python loops over pixels).

---

## 📄 License

MIT — free for personal and commercial use.
