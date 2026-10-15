"""MusicXML and two-staff, channel-zero listening MIDI export."""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mido


def prepare_midi_notes(notes):
    groups, seen = ({}, set())
    duplicates, simultaneous = (0, 0)
    for pitch, onset, duration, staff in notes:
        p, on, off, s = (
            int(pitch),
            round(onset * 480),
            round((onset + duration) * 480),
            int(staff),
        )
        if off <= on:
            raise ValueError("NOTE must have positive MIDI-tick duration")
        identity = (s, p, on, off)
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        starts = groups.setdefault((s, p), {})
        if on in starts:
            simultaneous += 1
            starts[on] = max(starts[on], off)
        else:
            starts[on] = off
    result, shortened = ([], 0)
    for (staff, pitch), starts in sorted(groups.items()):
        ordered = sorted(starts.items())
        for i, (on, off) in enumerate(ordered):
            if i + 1 < len(ordered) and off > ordered[i + 1][0]:
                off = ordered[i + 1][0]
                shortened += 1
            result.append((pitch, on, off, staff))
    return (
        result,
        dict(
            input_notes=len(notes),
            exact_duplicates_removed=duplicates,
            same_onset_merged=simultaneous,
            overlaps_shortened=shortened,
            export_policy="same_staff_pitch_dedup_clip_single_channel",
        ),
    )


def write_listening_midi(codec, score, path):
    if Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite MIDI: {path}")
    notes, changes = prepare_midi_notes(score.notes[score.notes[:, 2] > 0])
    changes["grace_skipped"] = int((score.notes[:, 2] == 0).sum())
    changes["raw_predicted_notes"] = len(score.notes)
    if any((note[3] not in (0, 1) for note in notes)):
        raise ValueError("Piano MIDI export supports only staff_0 and staff_1")
    staffs = (0, 1)
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    midi.tracks.append(meta)
    meta.append(mido.MetaMessage("set_tempo", tempo=500000))
    previous = 0
    metadata = []
    for start, meter in zip(score.bar_starts, score.meters):
        n, d = codec.meter(meter)
        tick = round(start * 480)
        metadata.append(
            (tick, mido.MetaMessage("time_signature", numerator=n, denominator=d))
        )
    names = (
        "Cb",
        "Gb",
        "Db",
        "Ab",
        "Eb",
        "Bb",
        "F",
        "C",
        "G",
        "D",
        "A",
        "E",
        "B",
        "F#",
        "C#",
    )
    last_key = None
    for start, fifths in zip(score.bar_starts, getattr(score, "keys", [])):
        if -7 <= fifths <= 7 and fifths != last_key:
            metadata.append(
                (
                    round(start * 480),
                    mido.MetaMessage("key_signature", key=names[int(fifths) + 7]),
                )
            )
            last_key = fifths
    for tick, message in sorted(metadata, key=lambda pair: pair[0]):
        meta.append(message.copy(time=tick - previous))
        previous = tick
    end = (
        round((score.bar_starts[-1] + score.bar_lengths[-1]) * 480)
        if len(score.bar_starts)
        else 0
    )
    meta.append(mido.MetaMessage("end_of_track", time=max(0, end - previous)))
    staff_channels = {}
    for staff in staffs:
        channel = 0
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=f"staff_{staff}"))
        events = []
        staff_channels[str(staff)] = [channel]
        track.append(mido.Message("program_change", program=0, channel=channel))
        for pitch, on, off, s in notes:
            if s == staff:
                events.extend([(on, 1, pitch, channel), (off, 0, pitch, channel)])
        previous = 0
        for tick, kind, pitch, channel in sorted(events):
            track.append(
                mido.Message(
                    "note_on" if kind else "note_off",
                    channel=channel,
                    note=pitch,
                    velocity=64 if kind else 0,
                    time=tick - previous,
                )
            )
            previous = tick
    midi.save(path)
    written = mido.MidiFile(path)
    if (
        len(written.tracks) != 3
        or [track.name for track in written.tracks[1:]] != ["staff_0", "staff_1"]
        or any((not message.is_meta for message in written.tracks[0]))
        or any(
            (
                message.channel != 0
                for track in written.tracks[1:]
                for message in track
                if not message.is_meta
            )
        )
    ):
        raise ValueError("MIDI readback staff track/channel layout mismatch")
    read_count = sum(
        (
            m.type == "note_on" and m.velocity > 0
            for track in written.tracks
            for m in track
        )
    )
    if read_count != len(notes):
        raise ValueError("MIDI readback note count mismatch")
    return dict(
        channels=1,
        staff_tracks=len(staff_channels),
        **{"empty": True} if not notes else {},
        staff_channels=staff_channels,
        notes=read_count,
        **{"rests": len(score.rests)} if hasattr(score, "rests") else {},
        **changes,
    )


def write_musicxml(score, path):
    """Serialize 480-PPQ note times, split/tie across bars, preserve grace.

    Two parts mean right/left hand. Pitch spelling is canonical sharp spelling,
    not a prediction. Rests, stems, spelling, and voice accuracy are not scored.
    """

    def add(parent, tag, text=None, **attrs):
        child = ET.SubElement(parent, tag, attrs)
        if text is not None:
            child.text = str(text)
        return child

    root = ET.Element("score-partwise", version="3.1")
    parts = add(root, "part-list")
    for staff in range(2):
        add(add(parts, "score-part", id=f"P{staff + 1}"), "part-name", f"staff_{staff}")
    starts = np.rint(np.asarray(score.bar_starts) * 480).astype(int)
    lengths = np.rint(np.asarray(score.bar_lengths) * 480).astype(int)
    notes = np.asarray(score.notes)
    ends = np.rint((notes[:, 1] + notes[:, 2]) * 480).astype(int)
    origin = int(starts[0])
    starts = starts - origin
    ons = np.rint(notes[:, 1] * 480).astype(int) - origin
    ends = ends - origin
    if (ons < 0).any() or (lengths <= 0).any():
        raise ValueError("Invalid score extent for metric XML")
    meters = list(score.meters)
    while len(ends) and max(ends) > starts[-1] + lengths[-1]:
        starts = np.r_[starts, starts[-1] + lengths[-1]]
        lengths = np.r_[lengths, lengths[-1]]
        meters.append(meters[-1])
    names = [
        ("C", 0),
        ("C", 1),
        ("D", 0),
        ("D", 1),
        ("E", 0),
        ("F", 0),
        ("F", 1),
        ("G", 0),
        ("G", 1),
        ("A", 0),
        ("A", 1),
        ("B", 0),
    ]
    for staff in range(2):
        part = add(root, "part", id=f"P{staff + 1}")
        voices = np.asarray(getattr(score, "voices", np.zeros(len(notes), dtype=int)))
        voice_map = {
            int(v): i + 1 for i, v in enumerate(np.unique(voices[notes[:, 3] == staff]))
        }
        for b, (start, length) in enumerate(zip(starts, lengths)):
            measure = add(part, "measure", number=str(b + 1), implicit="yes")
            attrs = add(measure, "attributes")
            add(attrs, "divisions", 480)
            fifths = int(score.keys[min(b, len(score.keys) - 1)])
            if -7 <= fifths <= 7:
                add(add(attrs, "key"), "fifths", fifths)
            time = add(attrs, "time")
            meter = meters[b] if "/" in meters[b] else "4/4"
            numerator, denominator = meter.split("/")
            add(time, "beats", numerator)
            add(time, "beat-type", denominator)
            clef = add(attrs, "clef")
            add(clef, "sign", "G" if staff == 0 else "F")
            add(clef, "line", 2 if staff == 0 else 4)
            selected = np.flatnonzero(
                (notes[:, 3] == staff)
                & (ons < start + length)
                & ((ends > start) | (ends == ons) & (ons >= start))
            )
            selected = sorted(
                selected, key=lambda i: (max(ons[i], start), ends[i] != ons[i], i)
            )
            for i in selected:
                on, off = (
                    max(ons[i], start) - start,
                    min(ends[i], start + length) - start,
                )
                if on:
                    add(add(measure, "forward"), "duration", on)
                n = add(measure, "note", id=f"n{i}_b{b}")
                grace = ends[i] == ons[i]
                if grace:
                    add(n, "grace", slash="yes")
                pitch = add(n, "pitch")
                step, alter = names[int(notes[i, 0]) % 12]
                add(pitch, "step", step)
                if alter:
                    add(pitch, "alter", alter)
                add(pitch, "octave", int(notes[i, 0]) // 12 - 1)
                if not grace:
                    add(n, "duration", off - on)
                ties = []
                if ons[i] < start:
                    ties.append("stop")
                if ends[i] > start + length:
                    ties.append("start")
                for kind in ties:
                    add(n, "tie", type=kind)
                add(n, "voice", voice_map[int(voices[i])])
                if grace:
                    add(n, "type", "eighth")
                if ties:
                    notation = add(n, "notations")
                    for kind in ties:
                        add(notation, "tied", type=kind)
                cursor = on if grace else off
                if cursor:
                    add(add(measure, "backup"), "duration", cursor)
            occupied_end = max(
                (min(ends[i], start + length) - start for i in selected), default=0
            )
            if occupied_end < length:
                if occupied_end:
                    add(add(measure, "forward"), "duration", occupied_end)
                remaining = int(length - occupied_end)
                chunks = (
                    [remaining]
                    if selected or remaining < 2
                    else [remaining // 2, remaining - remaining // 2]
                )
                for chunk in chunks:
                    n = add(measure, "note", **{"print-object": "no"})
                    add(n, "rest")
                    add(n, "duration", chunk)
                    add(n, "voice", 1)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
