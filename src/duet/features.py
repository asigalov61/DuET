"""50 FPS performance features, including sustained notes and onset velocity."""

import numpy as np
import torch


class RollRenderer:
    fps = 50
    num_pitches = 88
    pitch_min = 21
    roll_velocity = True

    @staticmethod
    def _sounding_offsets(
        pitch, onset, offset, pedal_time, pedal_value, pedal_enabled, end_time
    ):
        release = np.asarray(offset, dtype=np.float64).copy()
        if pedal_enabled and len(pedal_time):
            up_times = pedal_time[pedal_value == 0]
            state = np.searchsorted(pedal_time, offset, side="right") - 1
            down = (state >= 0) & (pedal_value[np.maximum(state, 0)] > 0)
            for i in np.flatnonzero(down):
                next_up = np.searchsorted(up_times, offset[i], side="right")
                release[i] = max(
                    offset[i],
                    up_times[next_up] if next_up < len(up_times) else end_time,
                )
            for p in np.unique(pitch):
                slots = np.flatnonzero(pitch == p)
                slots = slots[np.argsort(onset[slots], kind="stable")]
                for i, nxt in zip(slots[:-1], slots[1:]):
                    release[i] = max(offset[i], min(release[i], onset[nxt]))
        return release

    def _render_roll(
        self,
        frame_times: np.ndarray,
        pitch: np.ndarray,
        onset: np.ndarray,
        offset: np.ndarray,
        pedal_time: np.ndarray,
        pedal_value: np.ndarray,
        pedal_enabled: bool,
        velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        num_frames = len(frame_times)
        held = np.zeros((self.num_pitches, num_frames), dtype=np.float32)
        onset_roll = np.zeros_like(held)
        sustained = np.zeros_like(held)
        velocity_onset = np.zeros_like(held) if self.roll_velocity else None
        if self.roll_velocity:
            if velocity is None:
                raise ValueError("roll_velocity requires MIDI velocity values")
            velocity = np.asarray(velocity, dtype=np.float64)
            if velocity.ndim != 1 or len(velocity) != len(pitch):
                raise ValueError("velocity must have one value per note")
            if not np.all(np.isfinite(velocity) & (velocity >= 0) & (velocity <= 127)):
                raise ValueError("velocity must contain finite MIDI values in [0, 127]")
        step = (
            float(frame_times[1] - frame_times[0]) if num_frames > 1 else 1.0 / self.fps
        )
        release = self._sounding_offsets(
            pitch,
            onset,
            offset,
            pedal_time,
            pedal_value,
            pedal_enabled,
            frame_times[-1] + step,
        )
        tolerance = (
            8 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(frame_times))))
        )
        sample_times = frame_times + tolerance
        start = np.searchsorted(sample_times, onset, side="left")
        end = np.searchsorted(sample_times, offset, side="left")
        sound_end = np.searchsorted(sample_times, release, side="left")
        onset_frame = np.rint((onset - frame_times[0]) / step).astype(np.int64)
        for i, (p, on, j0, j1, j2, event) in enumerate(
            zip(pitch, onset, start, end, sound_end, onset_frame)
        ):
            row = int(p) - self.pitch_min
            if not 0 <= row < self.num_pitches:
                continue
            held[row, int(j0) : int(j1)] = 1.0
            sustained[row, int(j0) : int(j2)] = 1.0
            if on >= frame_times[0] and 0 <= event < num_frames:
                onset_roll[row, int(event)] = 1.0
                if velocity_onset is not None:
                    velocity_onset[row, int(event)] = max(
                        velocity_onset[row, int(event)], velocity[i] / 127.0
                    )
        if pedal_enabled and len(pedal_time):
            idx = np.searchsorted(pedal_time, frame_times, side="right") - 1
            pedal_on = np.where(idx >= 0, pedal_value[np.maximum(idx, 0)], 0).astype(
                np.float32
            )
        else:
            pedal_on = np.zeros(num_frames, dtype=np.float32)
        avail = np.full(
            (1, num_frames),
            float(pedal_enabled and len(pedal_time) > 0),
            dtype=np.float32,
        )
        channels = [held, sustained, onset_roll]
        if velocity_onset is not None:
            channels.append(velocity_onset)
        return np.concatenate([*channels, pedal_on[None, :], avail], axis=0)


def build_window(data, start):
    """Render carry-in sound; tokens belong to [start-8, start+32)."""
    left = max(0.0, start - 8.0)
    end = start + 32.0
    pitch, onset, offset, velocity = (
        data["ref_" + k] for k in ("pitch", "onset", "offset", "vel")
    )
    duration = float(data["duration_sec"])
    renderer = RollRenderer()
    relative = np.arange(round((end - left) * 50), dtype=np.float64) / 50
    times = left + relative
    valid = (times >= 0) & (times < duration)
    pedal_order = np.argsort(data["pedal_time"], kind="stable")
    pedal_time = data["pedal_time"][pedal_order]
    pedal_value = data["pedal_value"][pedal_order]
    sounding = renderer._sounding_offsets(
        pitch, onset, offset, pedal_time, pedal_value, True, end
    )
    keep = (onset < end) & (sounding > left) & (pitch >= 0) & (pitch <= 127)
    roll = renderer._render_roll(
        times,
        pitch[keep],
        onset[keep],
        offset[keep],
        pedal_time,
        pedal_value,
        True,
        velocity=velocity[keep],
    )
    roll[:, ~valid] = 0
    ids = np.flatnonzero(keep & (onset >= left))
    ids = ids[np.argsort(onset[ids], kind="stable")]
    context = ~((onset[ids] >= start) & (onset[ids] < end))
    arrays = dict(
        roll=roll,
        frame_times_rel=relative.astype(np.float32),
        frame_valid=valid,
        frame_is_context=~(
            (relative >= start - left) & (relative < end - left) & valid
        ),
        note_pitch=pitch[ids].astype(np.int64),
        note_onset_rel=(onset[ids] - left).astype(np.float32),
        note_duration_sec=(offset[ids] - onset[ids]).astype(np.float32),
        note_velocity=velocity[ids].astype(np.float32) / 127,
        note_masked=np.zeros(len(ids), dtype=bool),
        note_mask=np.ones(len(ids), dtype=bool),
        note_is_context=context,
    )
    batch = {k: torch.from_numpy(v)[None] for k, v in arrays.items()}
    batch["query_note_slots"] = torch.arange(len(ids))[None]
    return batch, ids, left
