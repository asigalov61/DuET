"""GPU reservation estimate from the already-trimmed PCM audio."""
import math
import wave


def prepared_seconds(path):
    with wave.open(str(path), "rb") as audio:
        seconds = audio.getnframes() / audio.getframerate()
    if not 0 < seconds <= 240 + 1 / 44100:
        raise ValueError("Expected nonempty prepared audio of at most 240 seconds")
    return seconds


def estimate_gpu_duration(path, output):
    # Heuristic with startup margin: 15 seconds minimum, 60 for 240s audio.
    # Not a runtime guarantee; calibrate using runtime reports.
    # `output` is required to match the decorated inference function signature.
    return max(15, math.ceil(10 + min(prepared_seconds(path), 240) * 5 / 24))
