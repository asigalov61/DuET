import importlib.util
from pathlib import Path
import tempfile
import unittest
import wave

spec = importlib.util.spec_from_file_location("gpu_budget", Path(__file__).resolve().parents[1] / "demo/gpu_budget.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


class BudgetTests(unittest.TestCase):
    def test_measures_pcm_not_filename_or_original_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "original_3600_seconds.wav"
            for duration, expected in [(6, 15), (60, 23), (120, 35), (180, 48), (240, 60)]:
                with wave.open(str(path), "wb") as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(8000)
                    f.writeframes(bytes(duration * 8000 * 2))
                self.assertEqual(budget.estimate_gpu_duration(path, "unused"), expected)
            for duration in [0, 241]:
                with wave.open(str(path), "wb") as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(8000)
                    f.writeframes(bytes(duration * 8000 * 2))
                with self.assertRaises(ValueError):
                    budget.estimate_gpu_duration(path, "unused")
