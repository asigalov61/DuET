"""Optional Transkun v2 audio frontend (official packaged checkpoint only)."""

import importlib.metadata
import importlib.resources
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch


def load_audio_model(device="cpu"):
    """Load the pinned Transkun model once, including at ZeroGPU startup."""
    try:
        version = importlib.metadata.version("transkun")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Audio input requires pip install 'duet-score[audio]'."
        ) from exc
    if version != "2.0.1":
        raise RuntimeError(f"Expected transkun 2.0.1, found {version}.")
    import moduleconf

    resources = importlib.resources.files("transkun.pretrained")
    with (
        importlib.resources.as_file(resources / "2.0.conf") as conf_path,
        importlib.resources.as_file(resources / "2.0.pt") as weight_path,
    ):
        settings = moduleconf.parseFromFile(str(conf_path))["Model"]
        model = settings.module.TransKun(conf=settings.config)
        # Only the checkpoint bundled in the pinned official package is loaded.
        # Upstream checkpoints use pickle; do not accept arbitrary user paths.
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("best_state_dict", checkpoint.get("state_dict"))
        if state is None:
            raise ValueError("Transkun checkpoint contains no model state.")
        model.load_state_dict(state, strict=True)
        del checkpoint, state
        model.to(device).eval()
    return model


def audio_to_midi(audio, output, *, device="cpu", model=None):
    """Decode audio with FFmpeg and transcribe with Transkun 2.0.1 / model v2."""
    audio, output = Path(audio), Path(output)
    if not audio.is_file():
        raise FileNotFoundError(audio)
    if output.exists():
        raise FileExistsError(output)
    model = model if model is not None else load_audio_model(device)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Audio input requires FFmpeg on PATH.")
    from transkun.Data import writeMidi

    with tempfile.TemporaryDirectory(prefix="duet-audio-") as tmp:
        decoded = Path(tmp) / "decoded.wav"
        # Convert explicitly to 16-bit PCM, independent of source bit depth/codec.
        result = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(audio.resolve()),
                "-map",
                "0:a:0",
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(model.fs),
                str(decoded),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"FFmpeg audio decoding failed: {result.stderr.strip()}")
        with wave.open(str(decoded), "rb") as stream:
            channels = stream.getnchannels()
            samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        if not samples.size:
            raise ValueError("Audio contains no samples.")
        audio_tensor = torch.from_numpy(
            samples.reshape(-1, channels).astype(np.float32) / 32768.0
        ).to(device)
        with torch.inference_mode():
            notes = model.transcribe(
                audio_tensor,
                stepInSecond=None,
                segmentSizeInSecond=None,
                discardSecondHalf=False,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation: never overwrite an existing performance MIDI.
        with output.open("xb") as handle:
            writeMidi(notes).write(handle)
    return {"name": "transkun", "package_version": "2.0.1", "model_version": "2.0"}
