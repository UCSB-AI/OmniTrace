"""From captured attention to token traces and sentence-level attributions.

Inputs: A [T, L, N] last-row attention per step/layer over the N prompt positions, a Timeline.
Scoring signals a_t(i):
    attmean : mean over layers (and heads, already averaged at capture)
    attraw  : last layer only
Tracing (per generated token t): i*(t) = argmax_i a_t(i) over evidence positions; unit(i*) is the
trace, a_t(i*) its confidence. Two variants are recorded:
    global      : argmax over all evidence positions (text units + media units)
    <modality>  : argmax restricted to that modality's positions (v1 video behaviour)
Sentence attribution: v1 curation (POS weight x conf^gamma votes, run coherence, p_min, coverage)
over the traces of the sentence's tokens; selected time units are merged into [t0, t1] spans.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .curation import DEFAULT_CURATE_CFG, SourceCurationConfig, curate_sources_with_conf
from .timeline import MODALITY_NAMES, OTHER, TEXT, Timeline, merge_intervals

SIGNALS = ("attmean", "attraw")


def signal(A: torch.Tensor, name: str) -> torch.Tensor:
    """[T, L, N] -> [T, N]"""
    if name == "attmean":
        return A.mean(1)
    if name == "attraw":
        return A[:, -1]
    if name.startswith("layer"):                     # e.g. "layer15"
        return A[:, int(name[5:])]
    raise ValueError(name)


def unit_aggregates(A: torch.Tensor, tl: Timeline) -> Dict[str, np.ndarray]:
    """Per-step, per-layer attention mass/max on every unit + mass on non-evidence positions.

    unit_sum [T, L, U], unit_max [T, L, U], other_sum [T, L], unit_max_mean [T, U]  (float32). Enough to
    recompute unit-level tracing for any layer subset/calibration offline without the GPU.
    """
    T, L, N = A.shape
    U = len(tl.units)
    pu = torch.as_tensor(tl.pos_unit, dtype=torch.long)
    src = pu >= 0
    idx = pu[src].view(1, 1, -1).expand(T, L, -1)
    vals = A[:, :, src]
    usum = torch.zeros(T, L, U).scatter_add_(2, idx, vals)
    umax = torch.zeros(T, L, U).scatter_reduce_(2, idx, vals, reduce="amax", include_self=False)
    other = A[:, :, ~src].sum(-1)
    # max over a unit's positions of the layer-mean attention = the exact v1 `attmean` unit score
    ummean = torch.zeros(T, U).scatter_reduce_(1, pu[src].view(1, -1).expand(T, -1), A.mean(1)[:, src],
                                               reduce="amax", include_self=False)
    return {"unit_sum": usum.numpy(), "unit_max": umax.numpy(), "other_sum": other.numpy(),
            "unit_max_mean": ummean.numpy()}


def _argmax(scores: torch.Tensor, mask: torch.Tensor):
    s = scores.masked_fill(~mask, float("-inf"))
    i = int(torch.argmax(s))
    return i, float(scores[i])


def token_traces(S: torch.Tensor, tl: Timeline, modalities: List[int]) -> Dict[str, List[Dict[str, Any]]]:
    """S [T, N]. Returns {"global": [...], "<modality>": [...]} with one trace per generated token."""
    pm = torch.as_tensor(tl.pos_modality, dtype=torch.long)
    pu = tl.pos_unit
    masks = {"global": torch.as_tensor(tl.source_mask())}
    for m in modalities:
        masks[MODALITY_NAMES[m]] = pm == m
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in masks}
    for t in range(S.shape[0]):
        for k, mask in masks.items():
            if not bool(mask.any()):
                out[k].append({"unit": -1, "pos": -1, "w": 0.0})
                continue
            i, w = _argmax(S[t], mask)
            out[k].append({"unit": int(pu[i]), "pos": i, "w": w})
    return out


def attribute_sentences(sentences: List[Dict[str, Any]], traces: List[Dict[str, Any]], tl: Timeline,
                        cfg: SourceCurationConfig = DEFAULT_CURATE_CFG) -> List[Dict[str, Any]]:
    """Curate each sentence's traced units; report text units by label and time units as merged spans."""
    out = []
    for sent in sentences:
        ids, pos, conf = [], [], []
        for k, tag in zip(sent["token_idx"], sent["pos"]):
            if k < len(traces) and traces[k]["unit"] >= 0:
                ids.append(traces[k]["unit"]); pos.append(tag); conf.append(traces[k]["w"])
        selected = curate_sources_with_conf(ids, pos, conf, cfg) if ids else []
        text_units, spans_by_sid = [], {}
        for u in selected:
            unit = tl.units[u]
            if unit["t0"] is None:
                text_units.append(unit["label"])
            else:
                spans_by_sid.setdefault(unit["sid"], []).append((unit["t0"], unit["t1"]))
        out.append({
            "text": sent["text"],
            "units": [int(u) for u in selected],
            "text_sources": text_units,
            "spans": {sid: merge_intervals(iv) for sid, iv in spans_by_sid.items()},
        })
    return out
