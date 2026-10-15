"""Decoded note geometry; all times are quarter notes."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Score:
    notes: np.ndarray
    bar_starts: np.ndarray
    bar_lengths: np.ndarray
    meters: list[str]
    voices: np.ndarray
    keys: np.ndarray
