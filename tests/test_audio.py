"""Audio routing tests; optional dependencies and weights are not downloaded."""

import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from duet.audio import audio_to_midi
from duet.pipeline import DuET


class AudioTests(unittest.TestCase):
    def test_missing_dependency_is_actionable(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "input.wav"
            path.touch()
            with patch(
                "duet.audio.importlib.metadata.version",
                side_effect=importlib.metadata.PackageNotFoundError,
            ):
                with self.assertRaisesRegex(RuntimeError, r"duet-score\[audio\]"):
                    audio_to_midi(path, Path(d) / "out.mid")

    def test_audio_pipeline_preserves_intermediate_and_reports(self):
        model = object.__new__(DuET)
        model.device = "cpu"

        def frontend(audio, output, **kwargs):
            output.write_bytes(b"performance")
            return {"name": "transkun", "model_version": "2.0"}

        def score(midi, output, **kwargs):
            self.assertEqual(midi.read_bytes(), b"performance")
            output.mkdir()
            (output / "score.mid").write_bytes(b"score")
            report = {"status": "complete", "input_notes": 2}
            (output / "report.json").write_text(json.dumps(report))
            return report

        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "result"
            with (
                patch("duet.audio.audio_to_midi", side_effect=frontend),
                patch.object(model, "transcribe", side_effect=score),
            ):
                result = model.transcribe_audio("input.wav", root)
                self.assertEqual(result["status"], "complete")
                self.assertEqual((root / "score.mid").read_bytes(), b"score")
                self.assertEqual(
                    (root / "performance.mid").read_bytes(), b"performance"
                )
                self.assertEqual(
                    json.loads((root / "report.json").read_text())["frontend"]["name"],
                    "transkun",
                )
                self.assertFalse((root / "score_stage").exists())
                with self.assertRaises(FileExistsError):
                    model.transcribe_audio("input.wav", root)

    def test_score_failure_retains_performance_and_diagnostics(self):
        model = object.__new__(DuET)
        model.device = "cpu"

        def frontend(audio, output, **kwargs):
            output.write_bytes(b"performance")
            return {"name": "transkun"}

        def fail(midi, output, **kwargs):
            output.mkdir()
            (output / "report.json").write_text(
                json.dumps({"status": "failed", "windows": [1]})
            )
            raise ValueError("bad window")

        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "result"
            with (
                patch("duet.audio.audio_to_midi", side_effect=frontend),
                patch.object(model, "transcribe", side_effect=fail),
            ):
                with self.assertRaisesRegex(ValueError, "bad window"):
                    model.transcribe_audio("input.wav", root)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["stage"], "midi_to_score")
            self.assertEqual(report["windows"], [1])
            self.assertTrue((root / "performance.mid").exists())

    def test_frontend_failure_is_not_complete(self):
        model = object.__new__(DuET)
        model.device = "cpu"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "result"
            with patch(
                "duet.audio.audio_to_midi", side_effect=RuntimeError("decode failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "decode failed"):
                    model.transcribe_audio("input.wav", root)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["stage"], "audio_to_midi")
