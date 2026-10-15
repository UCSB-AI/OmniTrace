"""Temporal attribution metrics shared by the evaluation scripts.

Intervals are (start, end) in seconds. Set metrics compare the union of predicted spans with the
union of GT spans; "official" metrics follow NExT-GQA's eval_ground.py (hull of the prediction,
max over GT segments).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

Span = Tuple[float, float]


def _len(iv: List[Span]) -> float:
    return sum(b - a for a, b in iv)


def _inter(a: List[Span], b: List[Span]) -> float:
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        tot += max(0.0, e - s)
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def set_metrics(pred: List[Span], gt: List[Span]) -> Dict[str, float]:
    P, G, I = _len(pred), _len(gt), _inter(pred, gt)
    p = I / P if P else 0.0
    r = I / G if G else 0.0
    return {"time_p": p, "time_r": r, "time_f1": 2 * p * r / (p + r) if p + r else 0.0,
            "iou": I / (P + G - I) if P + G - I > 0 else 0.0}


def official_metrics(pred: List[Span], gt_segments: List[Span]) -> Dict[str, float]:
    """Hull of the prediction vs each GT segment; max over GT segments (NExT-GQA eval_ground)."""
    if not pred:
        return {"iop": 0.0, "iou_hull": 0.0}
    a, b = min(x[0] for x in pred), max(x[1] for x in pred)
    best_iou = best_iop = 0.0
    for (s, e) in gt_segments:
        inter = max(0.0, min(b, e) - max(a, s))
        union = max(b, e) - min(a, s)
        best_iou = max(best_iou, inter / union if union > 0 else 0.0)
        best_iop = max(best_iop, inter / (b - a) if b > a else 0.0)
    return {"iop": best_iop, "iou_hull": best_iou}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels.sum(), (1 - labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):                    # average ranks for ties
        m = scores == v
        ranks[m] = ranks[m].mean()
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def answer_sentence_ids(sentences: List[Dict], options: List[str], pred_idx) -> List[int]:
    """Sentence(s) that state the final answer: the "Answer: X" line plus any sentence that names the
    chosen option's text (e.g. "...most likely a windsock, which indicates wind direction").
    Fallback: the last sentence."""
    import re
    ids = {i for i, s in enumerate(sentences) if "answer" in s["text"].lower()}
    if pred_idx is not None and options and 0 <= pred_idx < len(options):
        opt = options[pred_idx].strip().lower()
        if len(opt) >= 3:
            pat = re.compile(r"\b" + re.escape(opt) + r"\b")
            ids |= {i for i, s in enumerate(sentences) if pat.search(s["text"].lower())}
    return sorted(ids) or ([len(sentences) - 1] if sentences else [])


def question_gt(q: Dict) -> Tuple[str, List[Span], List[Span], float]:
    """(sid, merged GT spans, raw GT segments, duration) with spans clipped to [0, duration].

    Data card (OT_data/README.md): 162 NExT-GQA spans end up to 0.19 s past the video end."""
    from .timeline import merge_intervals
    ev = q["evidence"][0]
    sid = ev["sid"]
    dur = float(next(s for s in q["sources"] if s["sid"] == sid)["duration"])
    segs = [(max(0.0, float(a)), min(float(b), dur)) for a, b in ev["spans"]]
    segs = [(a, b) for a, b in segs if b > a]
    return sid, [tuple(x) for x in merge_intervals(segs)], segs, dur


def unit_overlap_labels(units: List[Dict], sid: str, gt: List[Span], short_spans: bool = False) -> Tuple[List[int], np.ndarray]:
    """Time units of `sid` and their 0/1 gold labels: a unit is gold when at least half of it lies inside the spans.
    short_spans (MMAU audio, where evidence is often one word or a brief sound): also gold when the overlap is at
    least half of the spans that touch the unit, so a 0.4-s word inside or across 1-s units still marks them."""
    vids = [u["uid"] for u in units if u["sid"] == sid and u["t0"] is not None]
    labels = []
    for uid in vids:
        u = units[uid]
        d = u["t1"] - u["t0"]
        inter = _inter([(u["t0"], u["t1"])], gt) if d > 0 else 0.0
        pos = d > 0 and inter / d >= 0.5
        if short_spans and not pos and inter > 0:
            touch = sum(b - a for a, b in gt if a < u["t1"] and b > u["t0"])
            pos = inter >= 0.5 * min(d, touch)
        labels.append(1 if pos else 0)
    return vids, np.array(labels)


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")
