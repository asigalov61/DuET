"""Read performance MIDI without score annotations."""

from collections import defaultdict, deque
from pathlib import Path
import warnings
import mido
import numpy as np


def read_performance_midi(path: Path, condition: str = "refined") -> dict:
    """Read performance notes, skipping nonpositive/unclosed notes and orphan note-offs.

    Preserve FIFO pairing and valid notes; never modify the source MIDI.
    Other structural errors remain fatal.
    """
    mid = mido.MidiFile(str(path))
    if mid.type == 2 or mid.ticks_per_beat <= 0:
        raise ValueError(
            "Only synchronous type 0/1 MIDI with positive PPQ is supported"
        )
    events = []
    for track_id, track in enumerate(mid.tracks):
        tick = 0
        for order, message in enumerate(track):
            tick += message.time
            events.append((tick, track_id, order, message))
    events.sort(key=lambda item: item[:3])
    active = defaultdict(deque)
    notes = []
    cleanup = dict(nonpositive_duration_notes=0, unmatched_note_off_events=0)
    pedal_time, pedal_value = ([], [])
    pedal_state = {}
    tempo, previous_tick, seconds = (500000, 0, 0.0)
    tempo_at_tick = {}
    ambiguous = False
    for tick, track_id, _, message in events:
        seconds += mido.tick2second(tick - previous_tick, mid.ticks_per_beat, tempo)
        previous_tick = tick
        if message.type == "set_tempo":
            if tick in tempo_at_tick and tempo_at_tick[tick] != message.tempo:
                raise ValueError(f"Conflicting MIDI tempos at tick {tick}")
            if message.tempo <= 0:
                raise ValueError("MIDI tempo must be positive")
            tempo_at_tick[tick] = tempo = message.tempo
        elif message.type in {"note_on", "note_off"}:
            key = (track_id, message.channel, message.note)
            if message.type == "note_on" and message.velocity > 0:
                if active[key] and (not ambiguous):
                    warnings.warn(
                        "Overlapping same-pitch note-ons on one track/channel are ambiguous; pairing note-offs FIFO without dropping notes",
                        RuntimeWarning,
                    )
                    ambiguous = True
                note_id = len(notes)
                notes.append([message.note, seconds, None, message.velocity])
                active[key].append(note_id)
            else:
                if not active[key]:
                    cleanup["unmatched_note_off_events"] += 1
                    continue
                note_id = active[key].popleft()
                if seconds <= notes[note_id][1]:
                    cleanup["nonpositive_duration_notes"] += 1
                    continue
                notes[note_id][2] = seconds
        elif message.type == "control_change" and message.control == 64:
            pedal_state[message.channel] = message.value >= 64
            pedal_time.append(seconds)
            pedal_value.append(int(any(pedal_state.values())))
        elif message.type == "control_change" and message.control in {120, 123}:
            if any(
                (queue for key, queue in active.items() if key[1] == message.channel)
            ):
                raise ValueError(
                    "All-notes-off/all-sound-off with active notes is ambiguous; supply explicit note-offs"
                )
    unclosed = sum((len(queue) for queue in active.values()))
    cleanup["unclosed_note_on_events"] = unclosed
    cleanup["input_note_on_events"] = len(notes)
    notes = [note for note in notes if note[2] is not None]
    cleanup["retained_notes"] = len(notes)
    if (cleanup["nonpositive_duration_notes"]
            or cleanup["unmatched_note_off_events"] or unclosed):
        warnings.warn(
            f"Skipped invalid MIDI events in {path}: {cleanup}", RuntimeWarning
        )
    if not notes:
        raise ValueError("MIDI contains no valid notes")
    if len(pedal_state) > 1:
        warnings.warn(
            "Per-channel sustain controllers are combined with logical OR for the single pedal input",
            RuntimeWarning,
        )
    if any(pedal_state.values()):
        warnings.warn(
            "Sustain pedal remains down at MIDI end; no synthetic release was added",
            RuntimeWarning,
        )
    array = np.asarray(notes, dtype=np.float64)
    prefix = "ref" if condition == "refined" else "cln"
    result = {
        f"{prefix}_{field}": array[:, i].astype(
            np.int64 if field in {"pitch", "vel"} else np.float64
        )
        for i, field in enumerate(("pitch", "onset", "offset", "vel"))
    }
    result.update(
        pedal_time=np.asarray(pedal_time, dtype=np.float64),
        pedal_value=np.asarray(pedal_value, dtype=np.int8),
        duration_sec=max(float(array[:, 2].max()), max(pedal_time, default=0.0)),
        sample_id=Path(path).stem,
        midi_input_cleanup=cleanup,
    )
    return result
