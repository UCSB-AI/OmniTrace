"""Unified source timeline: every input position -> (modality, source unit), every unit -> label / time interval.

Units are the atoms of attribution:
  - text units: labelled character ranges of the user prompt (question, each option, instruction)
  - time units (video/audio): a time interval [t0, t1] on one media source (e.g. one Qwen video
    temporal grid = 2 sampled frames)
Positions that are not evidence (system prompt, chat template, special tokens, whitespace)
get unit -1 and modality OTHER; their attention mass is still recorded (sink analysis).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

OTHER, TEXT, VIDEO, AUDIO, IMAGE = 0, 1, 2, 3, 4
MODALITY_NAMES = {OTHER: "other", TEXT: "text", VIDEO: "video", AUDIO: "audio", IMAGE: "image"}


@dataclass
class Timeline:
    n_positions: int
    pos_unit: np.ndarray                      # [N] int32, unit index or -1
    pos_modality: np.ndarray                  # [N] int8
    units: List[Dict[str, Any]] = field(default_factory=list)
    # unit dict: {"uid", "modality", "label", "sid", "t0", "t1", "n_tokens"}

    @classmethod
    def empty(cls, n: int) -> "Timeline":
        return cls(n, np.full(n, -1, np.int32), np.zeros(n, np.int8), [])

    def add_unit(self, positions: np.ndarray, modality: int, label: str,
                 sid: Optional[str] = None, t0: Optional[float] = None, t1: Optional[float] = None) -> int:
        uid = len(self.units)
        self.pos_unit[positions] = uid
        self.pos_modality[positions] = modality
        self.units.append({"uid": uid, "modality": MODALITY_NAMES[modality], "label": label, "sid": sid,
                           "t0": t0, "t1": t1, "n_tokens": int(len(positions))})
        return uid

    def unit_modality(self) -> np.ndarray:
        inv = {v: k for k, v in MODALITY_NAMES.items()}
        return np.array([inv[u["modality"]] for u in self.units], np.int8)

    def source_mask(self) -> np.ndarray:
        return self.pos_unit >= 0

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"n_positions": self.n_positions, "n_units": len(self.units)}
        for m, name in MODALITY_NAMES.items():
            out[f"n_{name}_tokens"] = int((self.pos_modality == m).sum())
        return out


def merge_intervals(intervals: List[Tuple[float, float]], gap: float = 1e-6) -> List[List[float]]:
    """Merge touching/overlapping [t0, t1] intervals."""
    iv = sorted((float(a), float(b)) for a, b in intervals if b > a)
    out: List[List[float]] = []
    for a, b in iv:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [[round(a, 3), round(b, 3)] for a, b in out]
