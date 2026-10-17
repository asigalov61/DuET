"""Regression tests use only generated data and never download weights."""

import tempfile
import unittest
from pathlib import Path

import mido
import numpy as np
import torch

from duet.codec import Codec
from duet.export import prepare_midi_notes, write_listening_midi, write_musicxml
from duet.features import build_window
from duet.layers import TimePitchRotaryEmbedding, time_pitch_positions
from duet.midi import read_performance_midi
from duet.score import Score


class InferenceTests(unittest.TestCase):
    def test_exact_half_open_history_and_new_ownership(self):
        onset = np.array([23.9, 24.0, 31.999, 32.0, 32.0, 63.999, 64.0])
        data = dict(
            ref_onset=onset,
            ref_offset=onset + 10,
            ref_pitch=np.array([48, 50, 60, 64, 67, 72, 74]),
            ref_vel=np.full(7, 90),
            pedal_time=np.array([]),
            pedal_value=np.array([]),
            duration_sec=74.0,
        )
        batch, ids, left = build_window(data, 32.0)
        self.assertEqual(left, 24.0)
        np.testing.assert_array_equal(ids, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(
            batch["note_is_context"][0], [True, True, False, False, False]
        )
        self.assertEqual(batch["roll"].shape, (1, 354, 2000))
        # Carry-in pitch48 remains held although it has no note token.
        self.assertEqual(batch["roll"][0, 48 - 21, 0], 1.0)
        self.assertEqual(batch["note_onset_rel"][0, 0], 0.0)
        self.assertEqual(batch["note_onset_rel"][0, 2], 8.0)
        self.assertTrue(batch["frame_is_context"][0, 399])
        self.assertFalse(batch["frame_is_context"][0, 400])

    def test_trailing_padding_is_zero(self):
        data = dict(
            ref_onset=np.array([0.0]),
            ref_offset=np.array([0.05]),
            ref_pitch=np.array([60]),
            ref_vel=np.array([127]),
            pedal_time=np.array([]),
            pedal_value=np.array([]),
            duration_sec=0.05,
        )
        batch, _, _ = build_window(data, 0.0)
        self.assertEqual(int(batch["frame_valid"].sum()), 3)
        self.assertFalse(bool(batch["roll"][..., 3:].any()))
        self.assertEqual(batch["roll"][0, 3 * 88 + 60 - 21, 0], 1.0)

    def test_pitch_rope_null_space_and_relative_time(self):
        rope = TimePitchRotaryEmbedding(64)
        x = torch.ones(1, 8, 2, 64)
        pos = time_pitch_positions(
            torch.tensor([[0.0, 0.02]]),
            torch.tensor([[60, 61]]),
            valid=torch.tensor([[True, False]]),
        )
        y = rope.rotate(x, pos)
        self.assertTrue(torch.isfinite(y).all())
        self.assertFalse(bool(y[:, :, 1, 32:].any()))
        self.assertFalse(torch.equal(y[:, :, 0, :32], y[:, :, 1, :32]))

    def test_midi_cleanup_does_not_merge_hands(self):
        notes = np.array(
            [[60, 0, 2, 0], [60, 0, 1, 0], [60, 0, 2, 0], [60, 1, 2, 0], [60, 0, 3, 1]],
            float,
        )
        cleaned, stats = prepare_midi_notes(notes)
        self.assertEqual(stats["exact_duplicates_removed"], 1)
        self.assertEqual(stats["same_onset_merged"], 1)
        self.assertEqual(stats["overlaps_shortened"], 1)
        self.assertEqual(
            cleaned, [(60, 0, 480, 0), (60, 480, 1440, 0), (60, 0, 1440, 1)]
        )

    def test_export_tracks_and_cross_bar_xml(self):
        score = Score(
            np.array([[60, 3, 2, 0], [48, 0, 4, 1], [64, 2, 0, 0]], float),
            np.array([0.0, 4.0]),
            np.array([4.0, 4.0]),
            ["4/4", "4/4"],
            np.array([0, 1, 0]),
            np.array([0, 0]),
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "score.mid"
            report = write_listening_midi(Codec(), score, path)
            midi = mido.MidiFile(path)
            self.assertEqual(len(midi.tracks), 3)
            for i in (1, 2):
                self.assertEqual(
                    {
                        m.channel
                        for m in midi.tracks[i]
                        if m.type in ("note_on", "note_off")
                    },
                    {0},
                )
            self.assertEqual(report["grace_skipped"], 1)
            with self.assertRaises(FileExistsError):
                write_listening_midi(Codec(), score, path)
            xml = Path(d) / "score.musicxml"
            write_musicxml(score, xml)
            from music21 import converter

            parsed = converter.parse(xml).stripTies()
            actual = []
            for staff, part in enumerate(parsed.parts):
                for n in part.flatten().notes:
                    for pitch in n.pitches if n.isChord else [n.pitch]:
                        actual.append(
                            (
                                int(pitch.midi),
                                float(n.offset),
                                float(n.duration.quarterLength),
                                staff,
                            )
                        )
            self.assertEqual(sorted(actual), sorted(map(tuple, score.notes.tolist())))

    def test_invalid_input_notes_are_counted(self):
        with tempfile.TemporaryDirectory() as d:
            midi = mido.MidiFile()
            t = mido.MidiTrack()
            midi.tracks.append(t)
            t.extend(
                [
                    mido.Message("note_off", note=40),
                    mido.Message("note_on", note=60, velocity=80),
                    mido.Message("note_off", note=60),
                    mido.Message("note_on", note=64, velocity=80),
                    mido.Message("note_off", note=64, time=480),
                ]
            )
            p = Path(d) / "input.mid"
            midi.save(p)
            data = read_performance_midi(p)
            self.assertEqual(data["midi_input_cleanup"]["retained_notes"], 1)
            self.assertEqual(
                data["midi_input_cleanup"]["nonpositive_duration_notes"], 1
            )
            self.assertEqual(data["midi_input_cleanup"]["unmatched_note_off_events"], 1)


    def test_unclosed_notes_are_skipped_without_changing_source(self):
        with tempfile.TemporaryDirectory() as d:
            midi = mido.MidiFile()
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.extend([
                mido.Message("note_on", note=60, velocity=80),
                mido.Message("note_on", note=64, velocity=80),
                mido.Message("note_off", note=64, time=480),
                mido.Message("note_on", note=64, velocity=90),
            ])
            path = Path(d) / "input.mid"
            midi.save(path)
            original = path.read_bytes()
            with self.assertWarnsRegex(RuntimeWarning, "unclosed_note_on_events"):
                data = read_performance_midi(path)
            self.assertEqual(data["ref_pitch"].tolist(), [64])
            self.assertEqual(data["ref_onset"].tolist(), [0.0])
            self.assertEqual(data["ref_offset"].tolist(), [0.5])
            self.assertEqual(data["midi_input_cleanup"], {
                "nonpositive_duration_notes": 0,
                "unmatched_note_off_events": 0,
                "unclosed_note_on_events": 2,
                "input_note_on_events": 3,
                "retained_notes": 1,
            })
            self.assertEqual(path.read_bytes(), original)

    def test_all_unclosed_notes_still_reject_empty_input(self):
        with tempfile.TemporaryDirectory() as d:
            midi = mido.MidiFile()
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.append(mido.Message("note_on", note=60, velocity=80))
            path = Path(d) / "input.mid"
            midi.save(path)
            with self.assertWarnsRegex(RuntimeWarning, "unclosed_note_on_events"):
                with self.assertRaisesRegex(ValueError, "no valid notes"):
                    read_performance_midi(path)


if __name__ == "__main__":
    unittest.main()
