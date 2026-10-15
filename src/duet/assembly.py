"""Reconstruct global score time from input-owned window predictions."""

from collections import Counter
import numpy as np
from .score import Score


def assemble_records(codec, records):
    if not records:
        raise ValueError("No note records")
    ids = [r["input_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated input note ID")
    a = codec.attributes(np.array([r["row"] for r in records]))
    bars = np.array([r["bar"] for r in records], dtype=np.int64)
    if (bars < 0).any() or bars.max() > 100000:
        raise ValueError("Invalid whole-piece bar number")
    info = {}
    conflicts = 0
    for b in np.unique(bars):
        ii = np.flatnonzero(bars == b)
        needed = int(a["onset_tick"][ii].max())
        candidates = [
            (
                int(a["length_tick"][i]),
                int(a["numerator"][i]),
                int(a["denominator"][i]),
                int(a["key"][i]),
            )
            for i in ii
            if a["length_tick"][i] > needed
        ]
        if not candidates:
            raise ValueError("No predicted bar length contains all its notes")
        all_meta = {
            (
                int(a["length_tick"][i]),
                int(a["numerator"][i]),
                int(a["denominator"][i]),
                int(a["key"][i]),
            )
            for i in ii
        }
        conflicts += len(all_meta) > 1
        info[int(b)] = Counter(candidates).most_common(1)[0][0]
    starts = []
    lengths = []
    meters = []
    keys = []
    tick = 0
    inherited = 0
    last = info[min(info)]
    for b in range(int(bars.max()) + 1):
        if b in info:
            last = info[b]
            L, n, d, k = last
        else:
            _, n, d, k = last
            L = round(480 * 4 * n / d)
            inherited += 1
        starts.append(tick)
        lengths.append(L)
        meters.append(f"{n}/{d}")
        keys.append(k)
        tick += L
    onset = np.array(starts)[bars] + a["onset_tick"]
    notes = np.column_stack(
        (
            [r["pitch"] for r in records],
            onset / 480,
            a["duration_tick"] / 480,
            a["staff"],
        )
    )
    score = Score(
        notes,
        np.array(starts) / 480,
        np.array(lengths) / 480,
        meters,
        a["voice"].copy(),
        np.array(keys),
    )
    score.note_order = np.arange(len(records))
    return (
        score,
        dict(
            raw_notes=len(records),
            inherited_empty_bars=inherited,
            bar_metadata_conflicts=int(conflicts),
            bar_metadata_policy="majority_joint_tuple_among_lengths_containing_all_onsets",
            initial_score_origin_q=0.0,
            trailing_empty_bars="omitted",
            grace_order="input_order_not_notation_order",
        ),
    )
