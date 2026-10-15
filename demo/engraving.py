"""Headless MuseScore export, separate from GPU inference."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def engrave(folder):
    folder = Path(folder)
    from musescore_runtime import ensure_musescore, VERSION
    executable = ensure_musescore()
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["QT_QPA_PLATFORMTHEME"] = "generic"
    env["QT_STYLE_OVERRIDE"] = "Fusion"
    # Isolate preferences between jobs; no display, MIDI devices or synthesizer.
    env["XDG_CONFIG_HOME"] = str(folder / ".musescore-config")
    midi = folder / "score.mid"
    if not midi.is_file():
        raise FileNotFoundError("score.mid is required for MuseScore engraving")
    jobs = [{"in": str(midi), "out": [str(folder / "score.engraved.musicxml"), str(folder / "score.svg"), str(folder / "score.pdf")]}]
    job = folder / "engraving-job.json"
    job.write_text(json.dumps(jobs))
    command = [executable, "-platform", "offscreen", "-j", str(job)]
    if shutil.which("xvfb-run"):
        command = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24"] + command
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    (folder / "engraving.log").write_text(result.stdout + result.stderr)
    pages = score_pages(folder)
    output = folder / "score.engraved.musicxml"
    if result.returncode or not output.is_file() or not pages or not (folder / "score.pdf").is_file():
        print("MuseScore export failed: " + (result.stdout + result.stderr)[-4000:], flush=True)
        raise RuntimeError("MuseScore export failed; original score MIDI is preserved")
    output.replace(folder / "score.musicxml")
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text())
    report["engraving"] = {"engine": "MuseScore", "version": VERSION, "headless": True, "source": "score.mid", "midi_import_settings": "MuseScore 4 defaults (isolated preferences)", "pages": len(pages)}
    report_path.write_text(json.dumps(report, indent=2) + "\n")


def score_pages(folder):
    return sorted(Path(folder).glob("score*.svg"), key=lambda p: [int(n) if n.isdigit() else n for n in re.split(r"(\d+)", p.name)])
