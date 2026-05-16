"""
web_ui/app.py
-------------
Gradio-based local web UI for the ASCII / ANSI art converter.

Launch via:  python main.py ui
         or: python web_ui/app.py
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    import cv2
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _build_renderer(
    preset, cols, match_mode, chars, color_mode, dither,
    edge_boost, contrast, brightness, gamma, saturation,
    invert, bilateral, font_size,
):
    from ascii_converter.core.renderer import Renderer, RenderConfig
    from ascii_converter.utils.presets import get_preset
    import dataclasses

    base = get_preset(preset)
    overrides = dict(
        cols=int(cols), match_mode=match_mode, chars=chars,
        color_mode=color_mode, dither=dither,
        edge_boost=float(edge_boost), contrast=float(contrast),
        brightness=float(brightness), gamma=float(gamma),
        saturation=float(saturation), invert=bool(invert),
        bilateral=bool(bilateral), font_size=int(font_size),
    )
    base.update(overrides)
    valid = {f.name for f in dataclasses.fields(RenderConfig)}
    filtered = {k: v for k, v in base.items() if k in valid}
    cfg = RenderConfig(**filtered)
    return Renderer(cfg), cfg


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------

def convert_image(image, preset, cols, match_mode, chars, color_mode, dither,
                  edge_boost, contrast, brightness, gamma, saturation,
                  invert, bilateral, font_size):
    """Image → (HTML preview, plain text, PNG file path)."""
    if image is None:
        return "<p style='color:red'>No image provided.</p>", "", None
    try:
        renderer, cfg = _build_renderer(
            preset, cols, match_mode, chars, color_mode, dither,
            edge_boost, contrast, brightness, gamma, saturation,
            invert, bilateral, font_size,
        )
        bgr = _pil_to_bgr(image)
        frame = renderer.render(bgr)

        # HTML preview
        if cfg.color_mode == "none":
            plain_escaped = frame.to_plain_string().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            preview = (
                "<div style='background:#0a0a0a;padding:10px;border-radius:8px;"
                "overflow:auto;max-height:580px;width:100%;'>"
                f"<pre style='font-family:monospace;font-size:9px;line-height:1.2;"
                f"color:#cdd6f4;margin:0;white-space:pre;'>{plain_escaped}</pre></div>"
            )
        else:
            from ascii_converter.core.colorizer import Colorizer
            html_col = Colorizer("html")
            frame.fg_codes = html_col.fg_codes(frame.color)
            from ascii_converter.output.html_out import HtmlOutput
            inner_html = HtmlOutput._frame_to_html(frame)
            preview = (
                "<div style='background:#0a0a0a;padding:10px;border-radius:8px;"
                "overflow:auto;max-height:580px;width:100%;'>"
                f"<pre style='font-family:monospace;font-size:9px;line-height:1.2;"
                f"color:#cdd6f4;margin:0;white-space:pre;'>{inner_html}</pre></div>"
            )

        # Plain text
        plain = frame.to_plain_string()

        # PNG
        from ascii_converter.output.image_out import ImageOutput
        img_out = ImageOutput(font_size=cfg.font_size, font_path=cfg.font_path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name
        img_out.save(frame, png_path)

        return preview, plain, png_path

    except Exception as exc:
        log.exception("Image conversion failed")
        return f"<p style='color:red'>Error: {exc}</p>", "", None


def convert_video(video_file, preset, cols, match_mode, chars, color_mode, dither,
                  edge_boost, contrast, brightness, gamma, saturation,
                  invert, bilateral, font_size, fps, output_format, max_frames):
    """Video → (output file path, status message)."""
    if video_file is None:
        return None, "No video provided."
    try:
        renderer, cfg = _build_renderer(
            preset, cols, match_mode, chars, color_mode, dither,
            edge_boost, contrast, brightness, gamma, saturation,
            invert, bilateral, font_size,
        )
        from ascii_converter.video.processor import VideoProcessor
        from ascii_converter.video.exporter import VideoExporter

        proc = VideoProcessor(cfg=cfg, target_fps=float(fps), workers=0)
        mf = int(max_frames) if max_frames else None
        frames = proc.process_file(video_file, max_frames=mf)

        ext = {"MP4": ".mp4", "GIF": ".gif", "HTML": ".html"}[output_format]
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            out_path = f.name

        exporter = VideoExporter(fps=float(fps), font_size=cfg.font_size)

        if output_format == "MP4":
            exporter.export_mp4(frames, out_path, audio_source=video_file)
        elif output_format == "GIF":
            exporter.export_gif(frames, out_path)
        elif output_format == "HTML":
            from ascii_converter.core.colorizer import Colorizer
            from ascii_converter.output.html_out import HtmlOutput
            html_col = Colorizer("html")
            for fr in frames:
                fr.fg_codes = html_col.fg_codes(fr.color)
            HtmlOutput(fps=float(fps)).save(frames, out_path)

        return out_path, f"{len(frames)} frames converted → {output_format}"

    except Exception as exc:
        log.exception("Video conversion failed")
        return None, f"Error: {exc}"


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

def build_ui():
    import gradio as gr
    from ascii_converter.utils.presets import list_presets
    from ascii_converter.core.char_library import CHARSETS

    PRESETS = list_presets()
    CHARSETS_LIST = list(CHARSETS.keys())

    def param_widgets():
        """Shared rendering parameter widgets."""
        preset     = gr.Dropdown(PRESETS, value="default", label="Preset")
        cols       = gr.Slider(40, 300, value=120, step=1, label="Columns")
        match_mode = gr.Dropdown(["density","structure","hybrid","ml"], value="hybrid", label="Match Mode")
        chars      = gr.Dropdown(CHARSETS_LIST, value="standard", label="Character Set")
        color_mode = gr.Dropdown(["truecolor","ansi256","ansi16","html","none"], value="truecolor", label="Color Mode")
        dither     = gr.Dropdown(["floyd_steinberg","stucki","jarvis","bayer","random","none"], value="floyd_steinberg", label="Dithering")
        with gr.Row():
            edge_boost  = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Edge Boost")
            contrast    = gr.Slider(0.5, 3.0, value=1.0, step=0.05, label="Contrast")
        with gr.Row():
            brightness  = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Brightness")
            gamma       = gr.Slider(0.3, 3.0, value=1.0, step=0.05, label="Gamma")
        with gr.Row():
            saturation  = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Saturation")
            font_size   = gr.Slider(8, 24, value=14, step=1, label="Font Size (pt)")
        with gr.Row():
            invert      = gr.Checkbox(label="Invert Luminance", value=False)
            bilateral   = gr.Checkbox(label="Bilateral Filter", value=False)
        return [preset, cols, match_mode, chars, color_mode, dither,
                edge_boost, contrast, brightness, gamma, saturation,
                invert, bilateral, font_size]

    with gr.Blocks(
        title="ASCII Art Converter",
    ) as demo:

        gr.Markdown("""
        # ASCII / ANSI Art Converter
        *Production-quality image & video → ASCII art. Supports braille, block Unicode,
        truecolor ANSI, HoG-hybrid matching, dithering and more.*
        """)

        with gr.Tabs():

            # ----------------------------------------------------------------
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(type="pil", label="Input Image")
                        img_p = param_widgets()
                        img_btn = gr.Button("Convert", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        img_preview = gr.HTML(label="ASCII Preview")
                        img_plain   = gr.Textbox(label="Plain Text", lines=8)
                        img_png     = gr.File(label="Download PNG")

                img_inputs = [img_input] + img_p
                img_outputs = [img_preview, img_plain, img_png]
                
                img_btn.click(fn=convert_image, inputs=img_inputs, outputs=img_outputs)
                
                # Real-time updates
                for widget in img_inputs:
                    if hasattr(widget, "release") and isinstance(widget, gr.Slider):
                        widget.release(fn=convert_image, inputs=img_inputs, outputs=img_outputs)
                    else:
                        widget.change(fn=convert_image, inputs=img_inputs, outputs=img_outputs)

            # ----------------------------------------------------------------
            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_input = gr.Video(label="Input Video")
                        vid_p = param_widgets()
                        with gr.Row():
                            vid_fps  = gr.Slider(1, 60, value=12, step=1, label="FPS")
                            vid_fmt  = gr.Radio(["MP4","GIF","HTML"], value="MP4",
                                                label="Output Format")
                        vid_mf   = gr.Number(value=None, label="Max Frames (blank=all)",
                                             precision=0)
                        vid_btn  = gr.Button("Convert", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        vid_status = gr.Textbox(label="Status", lines=3, interactive=False)
                        vid_out    = gr.File(label="Download Output")

                vid_inputs = [vid_input] + vid_p + [vid_fps, vid_fmt, vid_mf]
                vid_outputs = [vid_out, vid_status]
                
                vid_btn.click(fn=convert_video, inputs=vid_inputs, outputs=vid_outputs)
                
                # Real-time updates for video (only on certain quick parameters)
                for widget in vid_inputs[1:]: # skip video input to avoid re-running on load automatically
                    if hasattr(widget, "release") and isinstance(widget, gr.Slider):
                        widget.release(fn=convert_video, inputs=vid_inputs, outputs=vid_outputs)
                    else:
                        widget.change(fn=convert_video, inputs=vid_inputs, outputs=vid_outputs)

            # ----------------------------------------------------------------
            with gr.Tab("Tips & Reference"):
                gr.Markdown("""
                ## Best Settings Guide

                | Goal | Preset | Cols | Mode | Charset | Dither |
                |------|--------|------|------|---------|--------|
                | Max quality | `high_detail` | 200 | `hybrid` | `unicode_block` | `floyd_steinberg` |
                | Fastest | `default` | 80 | `density` | `standard` | `none` |
                | Anime / cartoon | `anime` | 160 | `hybrid` | `braille` | `none` |
                | Retro terminal | `retro` | 80 | `density` | `standard` | `none` |
                | Pixel art look | `block` | 120 | `density` | `block` | `bayer` |
                | Artistic poster | `artistic` | 120 | `hybrid` | `extended` | `bayer` |

                ## Color Mode Reference

                | Mode | Bits | Best for |
                |------|------|---------|
                | `truecolor` | 24-bit | PNG, modern terminals |
                | `ansi256` | 8-bit | Compatible terminals |
                | `ansi16` | 4-bit | Retro / widest support |
                | `html` | 24-bit | HTML output |
                | `none` | — | Plain text |

                ## CLI Quick-Start
                ```bash
                # Install
                pip install -r requirements.txt

                # Image → terminal
                python main.py image photo.jpg --preset high_detail --cols 180

                # Image → PNG
                python main.py image photo.jpg --out result.png --cols 200

                # Video → MP4
                python main.py video clip.mp4 --out ascii.mp4 --fps 24 --preset artistic

                # Webcam live
                python main.py webcam --preset anime
                ```
                """)

    return demo


def launch_ui(port: int = 7860, share: bool = False):
    import gradio as gr
    demo = build_ui()

    css = """
    body { background: #0f0f17; }
    footer { display: none !important; }
    .gr-button-primary { background: linear-gradient(135deg, #7c3aed, #2563eb) !important; }
    """

    demo.launch(
        server_port=port, 
        share=share, 
        show_error=True, 
        inbrowser=True,
        theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate"),
        css=css
    )


if __name__ == "__main__":
    launch_ui()
