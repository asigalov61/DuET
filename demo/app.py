"""Audio-to-score Space. No weights or tokens are served as files."""

import sys
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import spaces
import gradio as gr
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engraving import engrave, score_pages
from gpu_budget import estimate_gpu_duration, prepared_seconds
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from duet import DuET

ROOT = Path(tempfile.gettempdir()) / "duet-demo-jobs"
ROOT.mkdir(exist_ok=True)
MAX_SECONDS = 240
MODEL_REPO = os.environ["MODEL_REPO"]
MODEL_REVISION = os.environ["MODEL_REVISION"]
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def validate_input(path):
    if not path.is_file() or path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("Input must be an audio file of at most 100 MB.")
    if path.suffix.lower() in {".mid", ".midi"}:
        raise ValueError("This demo accepts audio only. Please upload a piano recording.")
    with path.open("rb") as source:
        if source.read(4) == b"MThd":
            raise ValueError("This demo accepts audio only. Please upload a piano recording.")


def prepare_audio(path, folder):
    """Decode at most 240 seconds; source files remain unchanged."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode:
        raise ValueError("Cannot read this audio file.")
    value = json.loads(probe.stdout).get("format", {}).get("duration")
    duration = float(value) if value not in (None, "N/A") else None
    target = folder / "prepared.wav"
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-t",
            str(MAX_SECONDS),
            "-map",
            "0:a:0",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode:
        raise ValueError("Audio decoding failed.")
    return target, {
        "limit_seconds": MAX_SECONDS,
        "original_seconds": duration,
        "processed_seconds": prepared_seconds(target),
        "truncated": duration > MAX_SECONDS if duration is not None else None,
    }


# ZeroGPU needs CUDA models initialized at module scope, after importing spaces.
# Local/CPU runs remain available for tests and fallback deployment.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DUET_MODEL = None
AUDIO_MODEL = None
if os.environ.get("DUET_SKIP_MODEL_LOAD") != "1":
    from musescore_runtime import ensure_musescore
    ensure_musescore()
    from duet.audio import load_audio_model

    DUET_MODEL = DuET.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, device=DEVICE
    )
    AUDIO_MODEL = load_audio_model(DEVICE)


@spaces.GPU(duration=estimate_gpu_duration)
def infer(path, output):
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    DUET_MODEL.transcribe_audio(path, output, audio_model=AUDIO_MODEL)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return {"requested_seconds": estimate_gpu_duration(path, output),
            "inference_wall_seconds": time.monotonic() - started,
            "estimator": "max(15, ceil(10 + min(processed_audio_seconds, 240) * 5 / 24))"}


def render_page(job_id, page):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id or ""):
        return "", "No score yet."
    xml = ROOT / job_id / "result" / "score.musicxml"
    if not xml.is_file():
        return "", "Result expired. Run again."
    pages = score_pages(xml.parent)
    if not pages:
        engrave(xml.parent)
        pages = score_pages(xml.parent)
    total = len(pages)
    number = min(max(int(page or 1), 1), total)
    svg = pages[number - 1].read_text()
    # Fit MuseScore's print-sized SVG to the screen, preserving its viewBox.
    svg = re.sub(
        r"<svg\b[^>]*>",
        lambda match: re.sub(r'\s(?:width|height|style)="[^"]*"', "", match.group(0))[:-1]
        + ' style="display:block;width:100%;max-width:100%;height:auto;">',
        svg,
        count=1,
    )
    # Keep notation legible
    # independently of the surrounding Gradio/browser light or dark theme.
    preview = (
        '<div style="background:#fff;color:#000;color-scheme:light;'
        'width:100%;max-width:960px;min-width:0;box-sizing:border-box;margin:0 auto;padding:12px;border-radius:8px;">'
        f'{svg}</div>'
    )
    return preview, f"Page {number} / {total}"


def run(upload, progress=gr.Progress()):
    for old in ROOT.iterdir():
        if old.is_dir() and time.time() - old.stat().st_mtime > 86400:
            shutil.rmtree(old)
    job = uuid.uuid4().hex
    folder = ROOT / job
    folder.mkdir()
    try:
        progress(0, desc="Preparing input")
        if not upload:
            raise ValueError("Upload a piano audio file.")
        path = folder / ("input" + Path(upload).suffix.lower())
        shutil.copyfile(upload, path)
        validate_input(path)
        path, audio_info = prepare_audio(path, folder)
        progress(0.1, desc="Waiting for GPU / transcribing")
        gpu_info = infer(str(path), str(folder / "result"))
        if audio_info is not None:
            report_path = folder / "result/report.json"
            report = json.loads(report_path.read_text())
            report["audio_preprocessing"] = audio_info
            report["gpu_budget"] = gpu_info
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        progress(0.95, desc="Rendering score")
        out = folder / "result"
        try:
            svg, caption = render_page(job, 1)
        except Exception:
            svg, caption = (
                "",
                "Preview unavailable. Download MusicXML to open in a score editor.",
            )
        files = [
            str(out / name)
            for name in ("score.mid", "score.musicxml", "score.pdf", "report.json", "performance.mid")
            if (out / name).is_file()
        ]
        return (
            files,
            svg,
            caption,
            job,
            1,
            "Complete. Only the first 240 seconds of audio are processed.",
        )
    except Exception as exc:
        # Do not expose stack traces, server paths, or credentials through the UI.
        if isinstance(exc, (ValueError, subprocess.TimeoutExpired)):
            message = (
                str(exc)
                if isinstance(exc, ValueError)
                else "Audio decoding timed out."
            )
        else:
            message = "Transcription failed. Try a shorter piano recording."
        raise gr.Error(html.escape(message)) from None


with gr.Blocks(title="DuET — Piano to Score", delete_cache=(3600, 86400)) as demo:
    gr.Markdown(
        "# DuET — Piano to Score\nUpload a piano recording. "
        "Only the first 240 seconds are processed; anything beyond is automatically truncated."
    )
    upload = gr.File(label="Piano audio", type="filepath", file_count="single", file_types=["audio"])
    button = gr.Button("Generate score", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    downloads = gr.File(label="Downloads", file_count="multiple")
    job_id = gr.State("")
    page = gr.Number(value=1, minimum=1, precision=0, label="Score page")
    caption = gr.Textbox(label="Preview", interactive=False)
    preview = gr.HTML()
    button.click(
        run,
        [upload],
        [downloads, preview, caption, job_id, page, status],
        concurrency_limit=1,
        api_name="transcribe",
    )
    page.submit(render_page, [job_id, page], [preview, caption], concurrency_limit=1)

if __name__ == "__main__":
    demo.queue(max_size=4).launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(ROOT)],
        max_file_size="100mb",
        show_error=False,
    )
