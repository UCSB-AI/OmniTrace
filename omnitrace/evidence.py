"""Question-level frame evidence from saved traces, for any attribution setting.

A setting is a dict (or "k=v,k=v" string) with:
    layers   all | last | early | middle | late | L<k> (single layer k) | L<a>-<b> (layers a..b)
    unit     max | sum | mean      how a unit's tokens are pooled (sum == mean for equal-size video groups)
    contrast none | step           subtract each unit's mean score over decoding steps
    scope    video | global        units competing for each token (global adds the question/option text)
    vote     hard | soft | mass    hard: argmax unit; soft: share over units; mass: raw scores, no argmax
    pos      on | off              POS weight of the voting token
    gamma    0 | 1 | 2             confidence exponent (hard/soft)
    sents    all | answer | {i,..}  which response sentences contribute (a set = per-sentence attribution)
Evidence(frame) = sum over contributing response tokens of the token's vote for that frame.

`max` is exact for layers=all/last when per-position signals were saved (pos_attmean/pos_attraw),
otherwise the layer-mean of per-layer unit maxima.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .curation import POS_W
from .metrics import answer_sentence_ids
from .timeline import merge_intervals

PRESETS: Dict[str, str] = {
    "v1":        "layers=all,unit=max,contrast=none,scope=video,vote=hard,pos=on,gamma=2,sents=all",
    "simple":    "layers=late,unit=sum,contrast=none,scope=video,vote=soft,pos=off,gamma=0,sents=all",
    "mass":      "layers=late,unit=sum,contrast=none,scope=video,vote=mass,pos=off,gamma=0,sents=all",
    "best_rank": "layers=late,unit=max,contrast=none,scope=global,vote=soft,pos=off,gamma=2,sents=all",
    "soft_all":  "layers=all,unit=sum,contrast=none,scope=video,vote=soft,pos=off,gamma=0,sents=all",
    "soft_last": "layers=last,unit=sum,contrast=none,scope=video,vote=soft,pos=off,gamma=0,sents=all",
    # added after probe A (2026-10-10): mass in late layers was best; variants for the causal test
    "mass_answer":     "layers=late,unit=sum,contrast=none,scope=video,vote=mass,pos=off,gamma=0,sents=answer",
    "mass_all_layers": "layers=all,unit=sum,contrast=none,scope=video,vote=mass,pos=off,gamma=0,sents=all",
}
DEFAULTS = dict(layers="late", unit="sum", contrast="none", scope="video", vote="soft", pos="off",
                gamma=0, sents="all")
# The fixed attention rule (chosen 2026-10-10 on NExT-GQA video; kept for every model and modality):
# what `run_attribution.py --signals attention` computes online.
ATTENTION = "mass"
# Score -> label rule. tau chosen 2026-10-11 by mean per-question F1 against gold quotes on the MMDocRAG dev 200
# (scripts/label_stats.py: 1.25 .492 | 1.5 .502 | 1.75 .494 | 2 .467 | 2.5 .424 | 3 .384), then frozen for every
# modality and model. Labels are recomputable from the stored scores.
LABEL_TAU = 1.5
# Per-modality thresholds chosen on dev data are saved in configs/label_tau.json (written by scripts/tau.py).
# A modality with an entry uses it; a modality without one falls back to LABEL_TAU until its dev sweep has run.
TAU_FILE = Path(__file__).resolve().parents[1] / "configs" / "label_tau.json"
DEV_FRAC, DEV_SEED = 0.2, 0


def tag_files(causal_dir, tag: str) -> List[Path]:
    """Row files of exactly this tag (causal_<tag>.jsonl and its shards), not of tags that merely start with it
    (blank vs blank_labels)."""
    d = Path(causal_dir)
    return sorted([*d.glob(f"causal_{tag}.jsonl"), *d.glob(f"causal_{tag}.shard*.jsonl")])


def tau_table() -> Dict:
    return json.loads(TAU_FILE.read_text()) if TAU_FILE.exists() else {}


def run_modality(units) -> str:
    """Key of a sample in the tau file: video | audio | image_text (by its candidate units)."""
    mods = {u["modality"] for u in units if u.get("sid") is not None}
    return "video" if "video" in mods else "audio" if "audio" in mods else "image_text"


def modality_tau(modality: str, default: float = LABEL_TAU) -> float:
    e = tau_table().get(modality)
    return float(e["tau"]) if e else default


def label_keys(label_tau, label_taus, modality: str):
    """(result key, tau) pairs of a causal run: one "label" at the modality's tau, or "label@t" per swept t."""
    if label_taus:
        return [(f"label@{t:g}", float(t)) for t in label_taus]
    return [("label", float(label_tau) if label_tau is not None else modality_tau(modality))]


def in_dev(qid: str, frac: float = DEV_FRAC, seed: int = DEV_SEED) -> bool:
    """Seeded per-id dev membership (independent of which other ids exist, so every script agrees)."""
    return int(hashlib.sha1(f"{seed}:{qid}".encode()).hexdigest()[:8], 16) / 2 ** 32 < frac


def split_ids(ids, modality: str, which: str = "auto", frac: Optional[float] = None) -> set:
    """all | dev | test | auto. dev = the seeded fraction used to choose tau; test = the rest.
    auto = test when the modality's tau was chosen on such a dev split (the tau file says so), else all."""
    e = tau_table().get(modality) or {}
    if which == "auto":
        which = "test" if e.get("dev_frac") else "all"
    if which == "all":
        return set(ids)
    frac = frac if frac is not None else e.get("dev_frac") or DEV_FRAC
    dev = {i for i in ids if in_dev(i, frac, e.get("dev_seed", DEV_SEED))}
    return dev if which == "dev" else set(ids) - dev


def parse_setting(spec: str) -> Dict:
    if spec.startswith("ext:"):               # another method's per-unit scores (omnitrace/external.py)
        return {"ext": spec[4:], "sents": "all"}
    spec = PRESETS.get(spec, spec)
    cfg = dict(DEFAULTS)
    for kv in spec.split(","):
        if kv.strip():
            k, v = kv.split("=")
            cfg[k.strip()] = int(v) if k.strip() == "gamma" else v.strip()
    return cfg


def setting_name(cfg: Dict) -> str:
    if "ext" in cfg:
        return f"ext:{cfg['ext']}"
    for name, spec in PRESETS.items():
        if parse_setting(spec) == cfg:
            return name
    return ",".join(f"{k}={cfg[k]}" for k in DEFAULTS)


def layer_slice(name: str, L: int) -> slice:
    named = {"all": slice(0, L), "last": slice(L - 1, L), "early": slice(0, L // 3),
             "middle": slice(L // 3, 2 * L // 3), "late": slice(2 * L // 3, L)}
    if name in named:
        return named[name]
    if name.startswith("L"):
        a, _, b = name[1:].partition("-")
        return slice(int(a), int(b or a) + 1)
    raise ValueError(name)


class TraceData:
    """Arrays of one question's trace + the response structure from its record."""

    def __init__(self, arrays, rec: Dict, options: List[str]):
        self.usum = np.asarray(arrays["unit_sum"], np.float64)            # [T, L, U]
        self.umax = np.asarray(arrays["unit_max"], np.float64)
        self.units = rec["units"]
        self.qid = rec.get("id")
        self.ext = {}                          # baseline rows of this question (external.attach)
        U = len(self.units)
        if self.usum.shape[2] != U:
            raise ValueError("trace / record unit mismatch")
        self.ntok = np.array([max(u["n_tokens"], 1) for u in self.units], np.float64)
        self.is_video = np.array([u["modality"] == "video" for u in self.units])
        self.is_text = np.array([u["modality"] == "text" for u in self.units])
        self.vids = [u["uid"] for u in self.units if u["modality"] == "video"]
        self.tok_sent, self.tok_pos = {}, {}
        for si, s in enumerate(rec["sentences"]):
            for k, p in zip(s["token_idx"], s["pos"]):
                self.tok_sent[k], self.tok_pos[k] = si, p
        T = self.usum.shape[0]
        self.toks = sorted(k for k in self.tok_sent if k < T)
        self.n_sent = len(rec["sentences"])
        self.ans_sents = set(answer_sentence_ids(rec["sentences"], options, rec.get("pred_idx")))
        self.pos_exact = {}
        files = getattr(arrays, "files", list(arrays.keys()))
        if "pos_unit" in files:
            pu = np.asarray(arrays["pos_unit"])
            idx = [np.where(pu == u)[0] for u in range(U)]
            for name, key in (("all", "pos_attmean"), ("last", "pos_attraw")):
                if key in files:
                    P = np.asarray(arrays[key], np.float64)
                    self.pos_exact[name] = np.stack(
                        [P[:, ix].max(1) if len(ix) else np.zeros(P.shape[0]) for ix in idx], 1)
        if "all" not in self.pos_exact and "unit_max_mean" in files:
            # saved by default since 2026-10-10: the exact v1 unit score without position dumps
            self.pos_exact["all"] = np.asarray(arrays["unit_max_mean"], np.float64)

    def unit_scores(self, layers: str, unit: str, contrast: str) -> np.ndarray:
        L = self.usum.shape[1]
        if unit == "max" and layers in self.pos_exact:
            S = self.pos_exact[layers].copy()
        else:
            sl = layer_slice(layers, L)
            S = (self.umax if unit == "max" else self.usum)[:, sl].mean(1)
            if unit == "mean":
                S = S / self.ntok[None, :]
        if contrast == "step":
            S = S - S.mean(0, keepdims=True)
        return S

    def evidence(self, cfg: Dict) -> np.ndarray:
        """Evidence per unit (length U); only video entries are meaningful for temporal use."""
        if "ext" in cfg:
            return self.external_evidence(cfg)
        S = self.unit_scores(cfg["layers"], cfg["unit"], cfg["contrast"])
        cand = self.is_video | (self.is_text if cfg["scope"] == "global" else False)
        sents = cfg["sents"]
        if sents == "all":
            sents = set(range(self.n_sent))
        elif sents == "answer":
            sents = self.ans_sents
        else:                                   # explicit sentence indices (per-sentence attribution)
            sents = set(sents)
        V = np.zeros(S.shape[1])
        for t in self.toks:
            if self.tok_sent[t] not in sents:
                continue
            w = POS_W.get(self.tok_pos[t], 0.3) if cfg["pos"] == "on" else 1.0
            if cfg["vote"] == "mass":
                V += w * np.where(cand, np.clip(S[t], 0, None), 0.0)
                continue
            s = np.where(cand, S[t], -np.inf)
            u = int(np.argmax(s))
            c = float(s[u])
            if not np.isfinite(c) or c <= 0:
                continue
            w *= c ** cfg["gamma"] if cfg["gamma"] else 1.0
            if cfg["vote"] == "hard":
                V[u] += w
            else:
                p = np.clip(np.where(cand, S[t], 0.0), 0, None)
                V += w * p / p.sum()
        return V


    def external_evidence(self, cfg: Dict) -> np.ndarray:
        """A baseline's scores (kind=scores): response vector for sents=all, else the sum of the chosen
        sentences' vectors (the attention scores are additive over sentences in the same way)."""
        row = self.ext.get(cfg["ext"])
        if row is None:
            raise KeyError(f"baseline '{cfg['ext']}' has no row for {self.qid}")
        if row.get("kind", "scores") != "scores":
            raise ValueError(f"baseline '{cfg['ext']}' is a set, not scores: use --ext_sets")
        sents = cfg["sents"]
        if sents == "all":
            return np.asarray(row["response"], np.float64)
        sents = self.ans_sents if sents == "answer" else set(sents)
        V = np.zeros(len(self.units))
        for j in sents:
            if j < len(row["sentences"]):
                V += np.asarray(row["sentences"][j], np.float64)
        return V


def candidate_groups(units: List[Dict]) -> Dict[str, List[int]]:
    """Attributable units by modality: time units and quotes, i.e. units that belong to a source (`sid`).
    Question / option / instruction text units are prompt, not evidence."""
    groups: Dict[str, List[int]] = {}
    for u in units:
        if u.get("sid") is not None:
            groups.setdefault(u["modality"], []).append(u["uid"])
    return groups


def select_units(scores, units: List[Dict], tau: float = LABEL_TAU) -> List[int]:
    """Score -> label rule, the same for every modality: a unit is selected when its score is at least
    `tau` times the mean score of the OTHER units it competes with. Units compete within their modality
    (video groups, audio seconds, text quotes, image quotes), so text cannot crowd out images; a unit
    with no same-modality peer competes with all candidates. Nothing selected = the sentence abstains."""
    groups = candidate_groups(units)
    everyone = [u for g in groups.values() for u in g]
    out = []
    for g in groups.values():
        pool = g if len(g) > 1 else everyone
        total = float(sum(scores[u] for u in pool))
        for u in g:
            v = float(scores[u])
            if v > 0 and (len(pool) == 1 or v >= tau * (total - v) / (len(pool) - 1)):
                out.append(u)
    return sorted(out)


def format_labels(selected: List[int], units: List[Dict]) -> Dict:
    """Readable labels: time units merged into [t0, t1] spans per source, other units by their label."""
    spans: Dict[str, list] = {}
    sources = []
    for u in selected:
        unit = units[u]
        if unit["t0"] is None:
            sources.append(unit["label"])
        else:
            spans.setdefault(unit["sid"], []).append((unit["t0"], unit["t1"]))
    return {"units": [int(u) for u in selected], "sources": sources,
            "spans": {sid: merge_intervals(iv) for sid, iv in spans.items()}}


def attention_attribution(unit_sum, sentences: List[Dict], units: Optional[List[Dict]] = None,
                          tau: float = LABEL_TAU) -> Dict:
    """The fixed attention attribution of one response, from unit_sum [T, L, U] (heads already averaged).

    token score     a_t(u) = mean over the late layers (last third) of the attention mass on unit u
    sentence score  sum of a_t(u) over the sentence's generated tokens
    response score  sum over all sentence tokens (= sum of the sentence scores)
    Scores are raw mass for EVERY unit (index = uid; text and media, no argmax, no renormalisation).
    They equal TraceData.evidence with the ATTENTION preset on the candidate units.
    With `units`, readable labels (select_units -> spans / source ids) are added at both levels.
    """
    cfg = parse_setting(ATTENTION)
    A = np.asarray(unit_sum, np.float64)
    T, L, U = A.shape
    sl = layer_slice(cfg["layers"], L)
    S = np.clip(A[:, sl].mean(1), 0, None) if T else np.zeros((0, U))
    fmt = lambda v: [float(f"{x:.6g}") for x in v]

    def entry(v, **extra):
        d = {**extra, "scores": fmt(v)}
        if units is not None:
            d.update(format_labels(select_units(v, units, tau), units))
        return d

    sents, used = [], set()
    for s in sentences:
        ks = [k for k in s["token_idx"] if k < T]
        used.update(ks)
        sents.append(entry(S[ks].sum(0) if ks else np.zeros(U), text=s["text"]))
    return {"setting": f"{ATTENTION}: layers={cfg['layers']},unit={cfg['unit']},vote={cfg['vote']}", "heads": "mean",
            "layers": [sl.start, sl.stop - 1], "label_tau": tau if units is not None else None, "sentences": sents,
            "response": entry(S[sorted(used)].sum(0) if used else np.zeros(U))}


def predicted_units(td: "TraceData", tau: float = LABEL_TAU, sents="all") -> List[int]:
    """Units the fixed setting labels for the response (sents="all") or for the given sentence set."""
    return select_units(td.evidence({**parse_setting(ATTENTION), "sents": sents}), td.units, tau)


def rank_frames(td: TraceData, cfg: Dict) -> Tuple[List[int], np.ndarray]:
    """Video unit ids sorted by evidence (desc; ties by time) and their scores."""
    V = td.evidence(cfg)
    vids = td.vids
    order = sorted(range(len(vids)), key=lambda i: (-V[vids[i]], i))
    return [vids[i] for i in order], V[[vids[i] for i in order]]


def take_budget(ranked: List[int], units: List[Dict], duration: float, frac: float) -> List[int]:
    """Top-ranked units until their total duration reaches frac * duration (at least one unit)."""
    out, cov = [], 0.0
    for u in ranked:
        if out and cov >= frac * duration - 1e-9:
            break
        out.append(u)
        cov += units[u]["t1"] - units[u]["t0"]
    return out
