"""Private audio/MIDI-to-score Space. No weights or tokens are served as files."""

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
from urllib.parse import parse_qs, urlparse

import spaces
import gradio as gr
import mido
import torch
import verovio
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


def youtube_id(url):
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port:
        raise ValueError("Use a normal HTTPS YouTube video link.")
    host = parsed.hostname
    if host == "youtu.be":
        value = parsed.path.strip("/")
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            value = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = parsed.path.strip("/").split("/")
            value = (
                parts[1] if len(parts) == 2 and parts[0] in {"shorts", "embed"} else ""
            )
    else:
        raise ValueError("Only YouTube video links are supported.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        raise ValueError("Invalid YouTube video ID.")
    return value


def download_youtube(url, folder):
    # Rebuild URL from an ID. Never pass arbitrary URLs, local paths or options.
    canonical = "https://www.youtube.com/watch?v=" + youtube_id(url)
    result = subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "--no-config",
            "--no-progress",
            "--socket-timeout",
            "20",
            "--retries",
            "1",
            "--max-filesize",
            "100M",
            "--match-filter",
            "!is_live",
            "--download-sections",
            f"*0-{MAX_SECONDS}",
            "-f",
            "bestaudio/best",
            "-o",
            str(folder / "youtube.%(ext)s"),
            canonical,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    candidates = [
        p for p in folder.glob("youtube.*") if p.suffix not in {".part", ".ytdl"}
    ]
    if result.returncode or len(candidates) != 1:
        raise ValueError(
            "YouTube download unavailable. "
            "Upload an authorized audio file instead. Login/DRM restrictions are not bypassed."
        )
    return candidates[0]


def validate_input(path, midi):
    if not path.is_file() or path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("Input must be a file of at most 100 MB.")
    if midi:
        if path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError("MIDI limit is 5 MB.")
        duration = mido.MidiFile(path).length
        if not 0 < duration <= 120:
            raise ValueError(
                "MIDI demo limit is 120 seconds; use the library for full pieces."
            )


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
        "truncated": duration > MAX_SECONDS if duration is not None else None,
    }


# ZeroGPU needs CUDA models initialized at module scope, after importing spaces.
# Local/CPU runs remain available for tests and fallback deployment.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DUET_MODEL = None
AUDIO_MODEL = None
if os.environ.get("DUET_SKIP_MODEL_LOAD") != "1":
    from duet.audio import load_audio_model

    DUET_MODEL = DuET.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, device=DEVICE
    )
    AUDIO_MODEL = load_audio_model(DEVICE)


@spaces.GPU(duration=180)
def infer(path, output, midi):
    if midi:
        return DUET_MODEL.transcribe(path, output)
    return DUET_MODEL.transcribe_audio(path, output, audio_model=AUDIO_MODEL)


def render_page(job_id, page):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id or ""):
        return "", "No score yet."
    xml = ROOT / job_id / "result" / "score.musicxml"
    if not xml.is_file():
        return "", "Result expired. Run again."
    toolkit = verovio.toolkit(False)
    toolkit.setResourcePath(str(Path(verovio.__file__).parent / "data"))
    toolkit.setOptions({"pageWidth": 1600, "pageHeight": 2200, "scale": 40})
    if not toolkit.loadFile(str(xml)):
        return "", "Preview unavailable; download MusicXML to view in a score editor."
    total = toolkit.getPageCount()
    number = min(max(int(page or 1), 1), max(total, 1))
    return toolkit.renderToSVG(number), f"Page {number} / {total}"


def run(source, upload, url, consent, progress=gr.Progress()):
    if not consent:
        raise gr.Error("Please confirm you have permission to process this input.")
    for old in ROOT.iterdir():
        if old.is_dir() and time.time() - old.stat().st_mtime > 86400:
            shutil.rmtree(old)
    job = uuid.uuid4().hex
    folder = ROOT / job
    folder.mkdir()
    try:
        progress(0, desc="Preparing input")
        if source == "YouTube":
            path = download_youtube(url, folder)
            midi = False
        else:
            if not upload:
                raise ValueError("Upload an audio or MIDI file.")
            midi = source == "MIDI"
            path = folder / (
                "input.mid" if midi else "input" + Path(upload).suffix.lower()
            )
            shutil.copyfile(upload, path)
        validate_input(path, midi)
        audio_info = None
        if not midi:
            path, audio_info = prepare_audio(path, folder)
            if source == "YouTube":
                audio_info["youtube_first_240_seconds_only"] = True
        progress(0.1, desc="Waiting for GPU / transcribing")
        infer(str(path), str(folder / "result"), midi)
        if audio_info is not None:
            report_path = folder / "result/report.json"
            report = json.loads(report_path.read_text())
            report["audio_preprocessing"] = audio_info
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        progress(0.95, desc="Rendering score")
        out = folder / "result"
        files = [
            str(out / name) for name in ["score.mid", "score.musicxml", "report.json"]
        ]
        if (out / "performance.mid").is_file():
            files.append(str(out / "performance.mid"))
        try:
            svg, caption = render_page(job, 1)
        except Exception:
            svg, caption = (
                "",
                "Preview unavailable. Download MusicXML to open in a score editor.",
            )
        return (
            files,
            svg,
            caption,
            job,
            1,
            (
                "Complete. Audio longer than 240 seconds is processed only for its first 240 seconds. "
                if not midi
                else "Complete. "
            )
            + "Temporary results expire after 24 hours.",
        )
    except Exception as exc:
        # Do not expose stack traces, server paths, or credentials through the UI.
        if isinstance(exc, (ValueError, subprocess.TimeoutExpired)):
            message = (
                str(exc)
                if isinstance(exc, ValueError)
                else "Download/decoding timed out."
            )
        else:
            message = "Transcription failed. Try a shorter piano clip or a MIDI file."
        raise gr.Error(html.escape(message)) from None


with gr.Blocks(title="DuET — Piano to Score", delete_cache=(3600, 86400)) as demo:
    gr.Markdown(
        "# DuET — Piano to Score\nUpload a piano recording or performance MIDI, or use a YouTube link. "
        "Download score MIDI and MusicXML, and preview the notation below.\n\n"
        "**Research demo · DuET weights: CC BY-NC 4.0 · Audio: first 240 seconds; MIDI: up to 120 seconds.** "
        "Audio uses Transkun v2. Longer audio is automatically truncated, not rejected. ZeroGPU usage counts toward your quota. Engraving is a preview, not a corrected edition."
    )
    source = gr.Radio(["Audio", "MIDI", "YouTube"], value="Audio", label="Input type")
    upload = gr.File(label="Audio or MIDI", type="filepath", file_count="single")
    url = gr.Textbox(
        label="YouTube URL", placeholder="https://www.youtube.com/watch?v=…"
    )
    consent = gr.Checkbox(
        label="I have permission to process this material for noncommercial use."
    )
    button = gr.Button("Generate score", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    downloads = gr.File(label="Downloads", file_count="multiple")
    job_id = gr.State("")
    page = gr.Number(value=1, minimum=1, precision=0, label="Score page")
    caption = gr.Textbox(label="Preview", interactive=False)
    preview = gr.HTML()
    button.click(
        run,
        [source, upload, url, consent],
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
