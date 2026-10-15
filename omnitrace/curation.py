"""Span-level source curation (CurateSourcesWithConf, v1 paper Alg. 1 / App. C).

From the first version of OmniTrace (arXiv 2604.13073v1); used only by the argmax-tracing comparison.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

# POS weights for generated tokens (Universal POS tags from spaCy)
POS_W: Dict[str, float] = {
    "NOUN": 1.0, "PROPN": 1.0, "NUM": 1.0,
    "VERB": 0.9, "ADJ": 0.8,
    "ADV": 0.5,
    "AUX": 0.1, "PRON": 0.1, "DET": 0.1, "ADP": 0.1,
    "CCONJ": 0.1, "SCONJ": 0.1, "PART": 0.1,
    "PUNCT": 0.0,
}

@dataclass(frozen=True)
class SourceCurationConfig:
    coverage: float = 0.85   # stop once selected sources cover this fraction of vote mass
    p_min: float = 0.07      # drop sources below this mass unless they have a strong run
    run_min: float = 0.10    # run_frac threshold that counts as a "strong run"
    alpha: float = 0.8       # score = alpha * mass + (1 - alpha) * longest-run mass
    gamma: float = 2.0       # confidence exponent
    # Ablation switches (all True = v1 default)
    use_pos: bool = True     # False -> every POS weight = 1
    use_conf: bool = True    # False -> every confidence = 1
    use_run: bool = True     # False -> alpha = 1 and no strong-run exemption
    use_pmin: bool = True    # False -> no p_min filtering

DEFAULT_CURATE_CFG = SourceCurationConfig()


def curate_sources_with_conf(
    source_ids: List[int],
    pos: List[str],
    conf: List[float],
    cfg: SourceCurationConfig = DEFAULT_CURATE_CFG,
) -> List[int]:
    """Pick a compact source set from per-token traces within one generated span.

    vote_t = POS_W[pos_t] * max(conf_t, 0) ** gamma
    p_mass[s]   = share of total vote traced to s
    run_frac[s] = largest contiguous run of votes for s / total
    rank by alpha * p_mass + (1 - alpha) * run_frac; skip weak sources (p_min, unless strong run);
    stop once cumulative p_mass >= coverage.
    """
    T = len(source_ids)
    if T == 0:
        return []
    if not (T == len(pos) == len(conf)):
        raise ValueError("source_ids, pos, conf must have the same length.")

    vote: List[float] = []
    for p, c in zip(pos, conf):
        pw = POS_W.get(p, 0.3) if cfg.use_pos else 1.0
        cw = (max(c, 0.0) ** cfg.gamma) if cfg.use_conf else 1.0
        vote.append(pw * cw)

    total = float(sum(vote))
    if total <= 0:
        return []

    mass = defaultdict(float)
    for s, v in zip(source_ids, vote):
        mass[s] += v
    p_mass = {s: m / total for s, m in mass.items()}

    run_max = defaultdict(float)
    cur_s, cur_run = source_ids[0], vote[0]
    for i in range(1, T):
        if source_ids[i] == cur_s:
            cur_run += vote[i]
        else:
            run_max[cur_s] = max(run_max[cur_s], cur_run)
            cur_s, cur_run = source_ids[i], vote[i]
    run_max[cur_s] = max(run_max[cur_s], cur_run)
    run_frac = {s: run_max[s] / total for s in mass.keys()}

    alpha = cfg.alpha if cfg.use_run else 1.0

    def score(s: int) -> float:
        return alpha * p_mass[s] + (1.0 - alpha) * run_frac[s]

    ranked = sorted(p_mass.keys(), key=score, reverse=True)

    selected: List[int] = []
    cum = 0.0
    for s in ranked:
        strong_run = cfg.use_run and run_frac[s] >= cfg.run_min
        if cfg.use_pmin and p_mass[s] < cfg.p_min and not strong_run:
            continue
        selected.append(s)
        cum += p_mass[s]
        if cum >= cfg.coverage:
            break

    return selected
