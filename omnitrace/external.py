"""Attributions from other methods, evaluated by the same causal tests as OmniTrace (plug-in interface).

Your method writes one row per question to <run>/baselines/<name>.jsonl:
    {"id", "method": name, "kind": "scores" | "set",
     "response": [U floats] | [uids],          scores: one value per unit uid (only candidate units are read)
     "sentences": [[U floats] | [uids], ...],  one entry per record sentence (same order as rec["sentences"])
     "cost": {...}}                             forward / backward passes, seconds (reported next to every score)
kind=scores  -> usable as a setting "ext:<name>" anywhere a preset is accepted (rank_frames / take_budget rows,
                per-sentence rows); TraceData.evidence reads it.
kind=set     -> a set of units without a ranking (self-citation). Evaluated like the label rows: hide exactly the
                set vs random units of the same count (key <name>), plus the attention rule's top units of the SAME
                count (key <setting>@<name>, size-matched comparison). Pass with --ext_sets <name>.
Probe rows (--probe loo | contextcite in the causal scripts) are the ablation forwards LOO / ContextCite are
fitted on; they are written under their own tag and never summarized.
"""
from __future__ import annotations

import json
import random
import zlib
from pathlib import Path
from typing import Dict, List

EXT_PREFIX = "ext:"


def ext_names(specs) -> List[str]:
    """Baseline names used by settings like 'ext:loo'."""
    return [s[len(EXT_PREFIX):] for s in specs or [] if s.startswith(EXT_PREFIX)]


def load_external(run_dir, names) -> Dict[str, Dict[str, Dict]]:
    """{name: {question id: row}} from <run>/baselines/<name>.jsonl (+ shard files)."""
    d = Path(run_dir) / "baselines"
    store = {}
    for n in names:
        files = sorted([*d.glob(f"{n}.jsonl"), *d.glob(f"{n}.shard*.jsonl")])
        if not files:
            raise FileNotFoundError(f"no baseline file {d}/{n}.jsonl (run the baseline first)")
        store[n] = {r["id"]: r for f in files for r in map(json.loads, f.open()) if r}
    return store


def attach(td, store: Dict[str, Dict[str, Dict]], qid: str) -> None:
    """Give a TraceData the baseline rows of its question (missing rows raise when the setting is used)."""
    td.ext = {n: rows[qid] for n, rows in store.items() if qid in rows}


def probe_variants(kind: str, units: List[int], qid: str, n: int = 32, keep_p: float = 0.5) -> Dict[str, tuple]:
    """Hidden-unit sets the perturbation baselines are fitted on.
    loo:         every candidate unit alone (one forward per unit).
    contextcite: n random ablations; each unit is KEPT with prob keep_p (ContextCite's default 0.5)."""
    if kind == "loo":
        return {f"probe|loo|{u}": (u,) for u in units}
    if kind == "contextcite":
        rng = random.Random(zlib.crc32(f"contextcite:{qid}".encode()))
        return {f"probe|cc|{i}": tuple(u for u in units if rng.random() >= keep_p) for i in range(n)}
    raise ValueError(kind)


def set_rows(td, names, all_v, variants, plan, sent_plan, rand_k, content_sents, match_settings, rank_frames,
             parse_setting, setting_name):
    """Variants for set-type baselines (see module doc). `match_settings`: specs whose top units of the same count
    are added as <setting>@<name> (e.g. mass@selfcite). Returns {name: {"response": k, "sent": {j: k}}}."""
    sizes = {}
    allset = set(all_v)
    for name in names:
        row = td.ext.get(name)
        if row is None:
            raise KeyError(f"baseline '{name}' has no row for {td.qid}")
        sizes[name] = {"response": 0, "sent": {}}

        def add(key, sel, sent=None):
            sel = sorted(set(sel) & allset)
            if not sel or len(sel) >= len(all_v):
                return len(sel)
            if sent is None:
                variants[f"{key}|del_top"] = tuple(sel)
                variants[f"{key}|keep_top"] = tuple(u for u in all_v if u not in set(sel))
                plan[key] = len(sel)
            else:
                variants[f"sent|{key}|{sent}"] = tuple(sel)
                sent_plan[f"{key}|{sent}"] = len(sel)
            rand_k(len(sel))
            return len(sel)

        k = add(name, row["response"])
        sizes[name]["response"] = k
        for spec in match_settings:
            if 0 < k < len(all_v):
                cfg = parse_setting(spec)
                add(f"{setting_name(cfg)}@{name}", rank_frames(td, cfg)[0][:k])
        for j in content_sents:
            sel = row["sentences"][j] if j < len(row["sentences"]) else []
            kj = add(name, sel, sent=j)
            sizes[name]["sent"][str(j)] = kj
            for spec in match_settings:
                if 0 < kj < len(all_v):
                    cfg = parse_setting(spec)
                    add(f"{setting_name(cfg)}@{name}", rank_frames(td, {**cfg, "sents": {j}})[0][:kj], sent=j)
    return sizes
