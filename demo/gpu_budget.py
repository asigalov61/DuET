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
    # Initial heuristic, not a measured runtime guarantee. Keep startup margin
    # and roughly the previous long-input budget; calibrate using runtime reports.
    # `output` is required to match the decorated inference function signature.
    return max(30, math.ceil(15 + 0.65 * prepared_seconds(path)))
