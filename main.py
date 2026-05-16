"""
main.py
-------
Command-line interface for the ASCII / ANSI art converter.

Usage examples
--------------
# Best-quality image to terminal
python main.py image photo.jpg --cols 160 --mode hybrid --color truecolor

# Image to PNG
python main.py image photo.jpg --out output.png --cols 200

# Image to HTML
python main.py image photo.jpg --out output.html --color html

# Video to terminal (live)
python main.py video clip.mp4 --fps 15 --cols 120

# Video to MP4
python main.py video clip.mp4 --out ascii_clip.mp4 --fps 24

# Video to GIF
python main.py video clip.mp4 --out ascii_clip.gif --fps 12

# Video to HTML animation
python main.py video clip.mp4 --out ascii_clip.html --fps 15

# Webcam live view
python main.py webcam --cols 100 --color truecolor

# List presets
python main.py presets

# Use a preset
python main.py image photo.jpg --preset anime

# Launch Gradio web UI
python main.py ui
"""

from __future__ import annotations

import logging
import os
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from ascii_converter.core.renderer import Renderer, RenderConfig
from ascii_converter.core.char_library import CHARSETS
from ascii_converter.core.dithering import DitherMethod
from ascii_converter.output.terminal import TerminalOutput
from ascii_converter.output.image_out import ImageOutput
from ascii_converter.output.html_out import HtmlOutput
from ascii_converter.output.gif_out import GifOutput
from ascii_converter.video.processor import VideoProcessor
from ascii_converter.video.exporter import VideoExporter
from ascii_converter.utils.presets import get_preset, list_presets

console = Console()

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

def _common_options(f):
    """Decorator that attaches shared rendering options to a Click command."""
    options = [
        click.option("--preset", default="default", show_default=True,
                     type=click.Choice(list_presets()), help="Style preset."),
        click.option("--cols", default=None, type=int, help="Character columns (overrides preset)."),
        click.option("--rows", default=None, type=int, help="Character rows (None=auto)."),
        click.option("--mode", default=None,
                     type=click.Choice(["density", "structure", "hybrid", "ml"]),
                     help="Character matching mode."),
        click.option("--chars", default=None,
                     type=click.Choice(list(CHARSETS.keys())),
                     help="Character set."),
        click.option("--color", "color_mode", default=None,
                     type=click.Choice(["none", "ansi16", "ansi256", "truecolor", "html"]),
                     help="Colour mode."),
        click.option("--dither", default=None,
                     type=click.Choice(["none","floyd_steinberg","stucki","jarvis","bayer","random"]),
                     help="Dithering algorithm."),
        click.option("--edge-boost", default=None, type=float, help="Edge enhancement strength [0-1]."),
        click.option("--contrast", default=None, type=float, help="Contrast multiplier."),
        click.option("--brightness", default=None, type=float, help="Brightness offset [-1, 1]."),
        click.option("--gamma", default=None, type=float, help="Gamma correction."),
        click.option("--saturation", default=None, type=float, help="Saturation multiplier."),
        click.option("--invert", is_flag=True, default=False, help="Invert luminance."),
        click.option("--bilateral", is_flag=True, default=False, help="Apply bilateral filter."),
        click.option("--font-path", default=None, help="Path to TTF monospace font."),
        click.option("--font-size", default=14, show_default=True, type=int, help="Font size in pt."),
        click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging."),
    ]
    for opt in reversed(options):
        f = opt(f)
    return f


def _build_config(
    preset: str,
    cols, rows, mode, chars, color_mode, dither,
    edge_boost, contrast, brightness, gamma, saturation,
    invert, bilateral, font_path, font_size, verbose,
    **extra,
) -> RenderConfig:
    """Merge preset + CLI overrides into a RenderConfig."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg_dict = get_preset(preset)

    # Apply CLI overrides (only non-None values)
    overrides = {
        "cols": cols,
        "rows": rows,
        "match_mode": mode,
        "chars": chars,
        "color_mode": color_mode,
        "dither": dither,
        "edge_boost": edge_boost,
        "contrast": contrast,
        "brightness": brightness,
        "gamma": gamma,
        "saturation": saturation,
        "invert": invert,
        "bilateral": bilateral,
        "font_path": font_path,
        "font_size": font_size,
    }
    for k, v in overrides.items():
        if v is not None and v is not False:
            cfg_dict[k] = v
    # Handle boolean flags
    if invert:
        cfg_dict["invert"] = True
    if bilateral:
        cfg_dict["bilateral"] = True

    # Build RenderConfig from merged dict (only valid fields)
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(RenderConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    return RenderConfig(**filtered)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option("1.0.0", prog_name="ascii-converter")
def cli():
    """
    ░█████╗░░██████╗░█████╗░██╗██╗    ░█████╗░░█████╗░███╗░░██╗██╗░░░██╗
    ██╔══██╗██╔════╝██╔══██╗██║██║    ██╔══██╗██╔══██╗████╗░██║██║░░░██║
    ███████║╚█████╗░██║░░╚═╝██║██║    ██║░░╚═╝██║░░██║██╔██╗██║╚██╗░██╔╝
    ██╔══██║░╚═══██╗██║░░██╗██║██║    ██║░░██╗██║░░██║██║╚████║░╚████╔╝░
    ██║░░██║██████╔╝╚█████╔╝██║██║    ╚█████╔╝╚█████╔╝██║░╚███║░░╚██╔╝░░
    ╚═╝░░╚═╝╚═════╝░░╚════╝░╚═╝╚═╝    ╚════╝░░╚════╝░╚═╝░░╚══╝░░░╚═╝░░░

    Production-quality ASCII/ANSI art converter for images & videos.
    """


# ---------------------------------------------------------------------------
# image command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--out", default=None, help="Output path (.txt/.png/.html). Default: terminal.")
@_common_options
def image(input_path, out, **kwargs):
    """Convert a single image to ASCII art."""
    import cv2

    cfg = _build_config(**kwargs)
    renderer = Renderer(cfg)

    img = cv2.imread(input_path)
    if img is None:
        console.print(f"[red]ERROR: Cannot read image: {input_path}[/red]")
        sys.exit(1)

    with console.status("[bold green]Converting image...[/bold green]"):
        frame = renderer.render(img)

    _dispatch_output_single(frame, out, cfg)


# ---------------------------------------------------------------------------
# video command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--out", default=None,
              help="Output path (.mp4/.gif/.html/.txt). Default: terminal live view.")
@click.option("--fps", default=None, type=float, help="Target output FPS.")
@click.option("--max-frames", default=None, type=int, help="Limit number of frames.")
@click.option("--workers", default=0, type=int,
              help="Parallel workers (0 = auto, uses all CPUs-1).")
@click.option("--temporal", default=0.0, type=float,
              help="Temporal blending factor [0-1] to reduce flicker.")
@click.option("--audio/--no-audio", default=True, help="Include audio in MP4 output.")
@_common_options
def video(input_path, out, fps, max_frames, workers, temporal, audio, **kwargs):
    """Convert a video file to ASCII art."""
    cfg = _build_config(**kwargs)
    proc = VideoProcessor(
        cfg=cfg,
        target_fps=fps,
        workers=workers,
        temporal=temporal,
        preview=False,
    )

    ext = os.path.splitext(out)[1].lower() if out else ""

    # For terminal live view we stream frame-by-frame
    if out is None:
        term = TerminalOutput(
            use_bg=cfg.use_bg_color,
            clear_between=True,
            fps_display=True,
        )
        term.clear_screen()
        term.hide_cursor()
        try:
            for _, frame in proc.frame_generator(input_path):
                term.display(frame)
        except KeyboardInterrupt:
            pass
        finally:
            term.show_cursor()
        return

    # Batch process all frames
    with console.status("[bold green]Processing video...[/bold green]"):
        frames = proc.process_file(input_path, max_frames=max_frames)

    audio_src = input_path if audio and ext == ".mp4" else None
    exporter = VideoExporter(
        fps=fps or 12.0,
        font_size=cfg.font_size,
        font_path=cfg.font_path,
    )

    if ext == ".mp4":
        exporter.export_mp4(frames, out, audio_source=audio_src)
    elif ext == ".gif":
        exporter.export_gif(frames, out)
    elif ext == ".html":
        _save_html(frames, out, cfg, fps=fps or 12.0)
    elif ext == ".txt":
        with open(out, "w", encoding="utf-8") as f:
            for fr in frames:
                f.write(fr.to_plain_string() + "\n\n")
        console.print(f"[green]Saved text → {out}[/green]")
    else:
        console.print(f"[red]Unknown output extension: {ext}[/red]")
        sys.exit(1)

    console.print(f"[bold green]Done![/bold green] Saved → [cyan]{out}[/cyan]")


# ---------------------------------------------------------------------------
# webcam command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--device", default=0, type=int, help="Webcam device index.")
@_common_options
def webcam(device, **kwargs):
    """Live ASCII art from webcam (press Ctrl-C to stop)."""
    cfg = _build_config(**kwargs)
    proc = VideoProcessor(cfg=cfg)
    term = TerminalOutput(
        use_bg=cfg.use_bg_color,
        clear_between=True,
        fps_display=True,
    )
    term.clear_screen()
    term.hide_cursor()

    def on_frame(frame):
        term.display(frame)

    try:
        console.print("[bold]Webcam mode — press Ctrl-C to stop[/bold]")
        proc.stream_webcam(device=device, callback=on_frame)
    finally:
        term.show_cursor()


# ---------------------------------------------------------------------------
# presets command
# ---------------------------------------------------------------------------

@cli.command("presets")
def cmd_presets():
    """List available style presets."""
    from ascii_converter.utils.presets import PRESETS

    table = Table(title="Available Style Presets", show_header=True, header_style="bold magenta")
    table.add_column("Preset", style="cyan", no_wrap=True)
    table.add_column("Key Settings", style="white")

    descriptions = {
        "default":      "Balanced defaults: 120 cols, hybrid, truecolor, Floyd-Steinberg",
        "high_detail":  "200 cols, Sobel edges, hybrid matching — best for complex scenes",
        "retro":        "80 cols, ASCII only, 16-colour ANSI — classic terminal look",
        "artistic":     "120 cols, Canny edges, Bayer dither — graphic novel vibe",
        "minimal":      "80 cols, space+10 chars, no colour — ultra-clean",
        "anime":        "160 cols, braille, vivid saturation — flat-colour anime",
        "matrix":       "100 cols, digit characters, green ANSI rain",
        "braille":      "160 cols, full braille charset, maximum density",
        "block":        "120 cols, half-block Unicode, highest colour fidelity",
    }

    for name in list_presets():
        table.add_row(name, descriptions.get(name, ""))

    console.print(table)


# ---------------------------------------------------------------------------
# UI command (Gradio)
# ---------------------------------------------------------------------------

@cli.command("ui")
@click.option("--port", default=7860, type=int, help="Gradio server port.")
@click.option("--share", is_flag=True, default=False, help="Create public Gradio share link.")
def cmd_ui(port, share):
    """Launch the Gradio web UI."""
    try:
        from web_ui.app import launch_ui
        launch_ui(port=port, share=share)
    except ImportError as e:
        console.print(f"[red]Gradio not installed: {e}[/red]")
        console.print("Install it with: pip install gradio")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dispatch_output_single(frame, out, cfg):
    if out is None:
        term = TerminalOutput(use_bg=cfg.use_bg_color)
        term.display(frame)
        return

    ext = os.path.splitext(out)[1].lower()
    if ext == ".png":
        img_out = ImageOutput(font_path=cfg.font_path, font_size=cfg.font_size)
        img_out.save(frame, out)
        console.print(f"[green]Saved PNG → {out}[/green]")
    elif ext == ".html":
        # For HTML we need html colour mode
        from ascii_converter.core.renderer import Renderer, RenderConfig
        html_cfg = RenderConfig(**{**cfg.__dict__, "color_mode": "html"})
        html_renderer = Renderer(html_cfg)
        import cv2
        # We can't re-read here; save current frame with swapped colorizer
        from ascii_converter.core.colorizer import Colorizer
        html_col = Colorizer("html")
        frame.fg_codes = html_col.fg_codes(frame.color)
        _save_html([frame], out, cfg)
        console.print(f"[green]Saved HTML → {out}[/green]")
    elif ext == ".txt":
        with open(out, "w", encoding="utf-8") as f:
            f.write(frame.to_plain_string())
        console.print(f"[green]Saved text → {out}[/green]")
    else:
        # Print to terminal by default
        term = TerminalOutput(use_bg=cfg.use_bg_color)
        term.display(frame)


def _save_html(frames, out, cfg, fps=12.0):
    # Re-generate HTML-mode colour codes
    from ascii_converter.core.colorizer import Colorizer
    html_col = Colorizer("html")
    for f in frames:
        f.fg_codes = html_col.fg_codes(f.color)
    html_out = HtmlOutput(fps=fps, font_size=10)
    html_out.save(frames, out)
    console.print(f"[green]Saved HTML → {out}[/green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
