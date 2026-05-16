"""
html_out.py
-----------
Export AsciiFrame(s) to a self-contained HTML file with:
  - Per-character <span> colour styling
  - Optional JS animation for multiple frames (video → HTML player)
  - Responsive layout, dark background, monospace font
  - No external dependencies (all inline)
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

from ascii_converter.core.renderer import AsciiFrame

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0a0a0a;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', 'Courier New', monospace;
  }}
  #controls {{
    margin: 12px 0;
    display: flex;
    gap: 10px;
    align-items: center;
  }}
  button {{
    background: #1e1e2e;
    border: 1px solid #444;
    color: #cdd6f4;
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    transition: background 0.2s;
  }}
  button:hover {{ background: #313244; }}
  #fps-label {{
    color: #6c7086;
    font-size: 13px;
  }}
  input[type=range] {{ accent-color: #89b4fa; width: 120px; }}
  #ascii-container {{
    overflow: auto;
    max-width: 100vw;
    max-height: 90vh;
    padding: 8px;
    background: #0a0a0a;
    border-radius: 4px;
  }}
  pre#ascii-art {{
    font-size: {font_size}px;
    line-height: 1.2;
    letter-spacing: 0.05em;
    white-space: pre;
    display: block;
  }}
</style>
</head>
<body>
{controls_html}
<div id="ascii-container">
  <pre id="ascii-art"></pre>
</div>
<script>
const FRAMES = {frames_json};
const FPS_DEFAULT = {fps};
let currentFrame = 0;
let fps = FPS_DEFAULT;
let playing = {autoplay};
let intervalId = null;

function renderFrame(idx) {{
  document.getElementById('ascii-art').innerHTML = FRAMES[idx];
}}

function startPlay() {{
  if (intervalId) clearInterval(intervalId);
  intervalId = setInterval(() => {{
    currentFrame = (currentFrame + 1) % FRAMES.length;
    renderFrame(currentFrame);
  }}, 1000 / fps);
}}

function stopPlay() {{
  if (intervalId) {{ clearInterval(intervalId); intervalId = null; }}
}}

// Controls
const playBtn = document.getElementById('playBtn');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const fpsRange = document.getElementById('fpsRange');
const fpsLabel = document.getElementById('fps-label');

if (playBtn) {{
  playBtn.addEventListener('click', () => {{
    playing = !playing;
    playBtn.textContent = playing ? '⏸ Pause' : '▶ Play';
    playing ? startPlay() : stopPlay();
  }});
}}
if (prevBtn) {{
  prevBtn.addEventListener('click', () => {{
    stopPlay(); playing = false;
    if (playBtn) playBtn.textContent = '▶ Play';
    currentFrame = (currentFrame - 1 + FRAMES.length) % FRAMES.length;
    renderFrame(currentFrame);
  }});
}}
if (nextBtn) {{
  nextBtn.addEventListener('click', () => {{
    stopPlay(); playing = false;
    if (playBtn) playBtn.textContent = '▶ Play';
    currentFrame = (currentFrame + 1) % FRAMES.length;
    renderFrame(currentFrame);
  }});
}}
if (fpsRange) {{
  fpsRange.addEventListener('input', () => {{
    fps = parseInt(fpsRange.value);
    fpsLabel.textContent = fps + ' FPS';
    if (playing) startPlay();
  }});
}}

// Initial render
renderFrame(0);
if (playing && FRAMES.length > 1) startPlay();
</script>
</body>
</html>
"""

_CONTROLS_MULTI = """
<div id="controls">
  <button id="prevBtn">⏮ Prev</button>
  <button id="playBtn">⏸ Pause</button>
  <button id="nextBtn">Next ⏭</button>
  <input type="range" id="fpsRange" min="1" max="60" value="{fps}">
  <span id="fps-label">{fps} FPS</span>
</div>
"""

_CONTROLS_SINGLE = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class HtmlOutput:
    """
    Converts one or more AsciiFrame objects to a self-contained HTML file.

    Parameters
    ----------
    font_size : CSS font-size in px for the <pre> element
    fps       : playback FPS when exporting video frames
    autoplay  : start playing automatically (multi-frame only)
    title     : HTML page title
    """

    def __init__(
        self,
        font_size: int = 10,
        fps: float = 12.0,
        autoplay: bool = True,
        title: str = "ASCII Art",
    ):
        self.font_size = font_size
        self.fps = fps
        self.autoplay = autoplay
        self.title = title

    # ------------------------------------------------------------------

    def save(self, frames: List[AsciiFrame], path: str) -> None:
        """
        Parameters
        ----------
        frames : one or more AsciiFrame objects
        path   : output .html file path
        """
        frame_htmls = [self._frame_to_html(f) for f in frames]
        frames_json = json.dumps(frame_htmls)

        multi = len(frames) > 1
        controls = (
            _CONTROLS_MULTI.format(fps=int(self.fps)) if multi else _CONTROLS_SINGLE
        )

        html = _HTML_TEMPLATE.format(
            title=self.title,
            font_size=self.font_size,
            frames_json=frames_json,
            fps=self.fps,
            autoplay="true" if (self.autoplay and multi) else "false",
            controls_html=controls,
        )

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        log.info("Saved HTML → %s  (%d frame(s))", path, len(frames))

    # ------------------------------------------------------------------

    @staticmethod
    def _frame_to_html(frame: AsciiFrame) -> str:
        """Convert a single AsciiFrame to an HTML string (no <pre> wrapper)."""
        lines = []
        for r in range(frame.rows):
            parts = []
            for c in range(frame.cols):
                ch = frame.chars[r, c]
                fg = frame.fg_codes[r, c]   # hex colour string from html mode
                # Escape HTML-special characters
                ch = ch.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                if fg and fg.startswith("#"):
                    parts.append(f'<span style="color:{fg}">{ch}</span>')
                else:
                    parts.append(ch)
            lines.append("".join(parts))
        return "\n".join(lines)
