"""Label threshold (tau) per modality: look up, choose on dev data, relabel a finished run (CPU).

    python scripts/tau.py show
    python scripts/tau.py get video                      # prints the saved value, or nothing if not chosen yet
    python scripts/tau.py forget audio                   # drop a saved value (e.g. the sweep was incomplete)
    python scripts/tau.py pick --modality video --run runs/nextgqa/qwen2.5-omni-7b --tag tau_blank --dev_frac 0.2
    python scripts/tau.py relabel --run runs/nextgqa/qwen2.5-omni-7b      # -> <run>/labels_tau<tau>.jsonl

Saved values live in configs/label_tau.json; every script that makes labels reads the modality's value from
there (omnitrace/evidence.py: modality_tau) and falls back to LABEL_TAU for a modality without an entry.

pick reads a label-only causal sweep on the dev questions (causal_*.py run --split dev --labels_only --no_sent
--label_taus ...; the launchers do this when the modality has no saved value). Per tau and question:
    score = [drop(hide the label) - drop(hide random units, same count)]            (deletion lift)
          + [drop(keep only random units, same count) - drop(keep only the label)]  (keep gain)
on the mean log-prob of the response tokens; a question whose label is empty (or is the whole source) scores 0.
The tau with the highest mean score is chosen, unless the default (LABEL_TAU, the value other modalities use)
is not reliably worse (paired bootstrap 95% interval of best - default includes 0): then the default is kept,
so that modalities share one value whenever the dev data do not argue against it.

relabel rewrites the readable labels of a finished run at the saved (or a given) tau from the stored scores:
no GPU and no new generation.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from omnitrace.evidence import (DEV_SEED, LABEL_TAU, TAU_FILE, attention_attribution, modality_tau, run_modality,
                                tag_files, tau_table)


def question_score(row, key):
    """(score, deletion lift, keep gain, labelled) of one tau on one question; no usable label -> zeros."""
    res, k = row["results"], row["plan"].get(key)
    if k is None:
        return 0.0, 0.0, 0.0, 0.0
    o = res["orig"]["resp_lp"]
    drop = lambda name: o - res[name]["resp_lp"]
    rand = [drop(n) for n in res if n.startswith(f"rand|{k}|")]
    keeprand = [drop(n) for n in res if n.startswith(f"keeprand|{k}|")]
    lift = drop(f"{key}|del_top") - float(np.mean(rand))
    gain = float(np.mean(keeprand)) - drop(f"{key}|keep_top")
    return lift + gain, lift, gain, 1.0


def pick(args):
    runp = Path(args.run)
    rows = [json.loads(l) for f in tag_files(runp / "causal", args.tag) for l in f.open() if l.strip()]
    rows = [r for r in {r["id"]: r for r in rows}.values() if np.isfinite(r["results"]["orig"]["resp_lp"])]
    if not rows:
        sys.exit(f"no rows for tag {args.tag} in {runp / 'causal'}")
    keys = sorted({k for r in rows for k in r["label_sizes"]}, key=lambda k: rows[0]["label_tau"].get(k, 0))
    taus = {k: next(r["label_tau"][k] for r in rows if k in r["label_tau"]) for k in keys}
    S = {k: np.array([question_score(r, k) for r in rows]) for k in keys}          # [n, 4]
    print(f"{len(rows)} dev questions ({runp}, tag {args.tag})")
    print(f"{'tau':>6s} {'score':>9s} {'del_lift':>9s} {'keep_gain':>9s} {'labelled%':>9s}")
    table = {}
    for k in keys:
        m = S[k].mean(0)
        table[f"{taus[k]:g}"] = {"score": float(m[0]), "del_lift": float(m[1]), "keep_gain": float(m[2]), "labelled": float(m[3])}
        print(f"{taus[k]:6.2f} {m[0]:9.4f} {m[1]:9.4f} {m[2]:9.4f} {100 * m[3]:9.1f}")
    best = max(keys, key=lambda k: S[k][:, 0].mean())
    chosen, why = best, "highest mean score"
    dflt = next((k for k in keys if abs(taus[k] - args.default) < 1e-9), None)
    if dflt is not None and dflt != best:
        d = S[best][:, 0] - S[dflt][:, 0]
        rng = np.random.default_rng(0)
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        print(f"best {taus[best]:g} vs default {taus[dflt]:g}: score difference {d.mean():.4f} [{lo:.4f}, {hi:.4f}]")
        if lo <= 0:
            chosen, why = dflt, f"default kept: {taus[best]:g} scores highest but not reliably higher (95% interval [{lo:.4f}, {hi:.4f}])"
    entry = {"tau": taus[chosen], "date": datetime.date.today().isoformat(), "why": why,
             "criterion": "causal, response level: deletion lift + keep gain of the label vs size-matched random "
                          "(mean log-prob of the response; empty label = 0)",
             "run": str(runp), "tag": args.tag, "n_dev": len(rows), "best_raw": taus[best], "table": table}
    if args.dev_frac:
        entry.update(dev_frac=args.dev_frac, dev_seed=DEV_SEED)
    tab = tau_table()
    if args.modality in tab and not args.force:
        sys.exit(f"{args.modality} already has tau = {tab[args.modality]['tau']} in {TAU_FILE} (use --force to replace)")
    tab[args.modality] = entry
    TAU_FILE.parent.mkdir(exist_ok=True)
    TAU_FILE.write_text(json.dumps(tab, indent=2) + "\n")
    print(f"{args.modality}: tau = {taus[chosen]:g} ({why}) -> {TAU_FILE}")


def relabel(args):
    runp = Path(args.run)
    recs = [json.loads(l) for f in sorted(runp.glob("records*.jsonl")) for l in f.open() if l.strip()]
    recs = [r for r in {r["id"]: r for r in recs}.values() if (runp / "traces" / r["meta"]["trace_file"]).exists()]
    if not recs:
        sys.exit(f"no records with traces in {runp}")
    tau = args.tau if args.tau is not None else modality_tau(run_modality(recs[0]["units"]))
    out = runp / f"labels_tau{tau:g}.jsonl"
    with out.open("w") as f:
        for r in sorted(recs, key=lambda r: r["id"]):
            z = np.load(runp / "traces" / r["meta"]["trace_file"])
            att = attention_attribution(z["unit_sum"], r["sentences"], r["units"], tau)
            f.write(json.dumps({"id": r["id"], "label_tau": tau, "response": att["response"],
                                "sentences": att["sentences"]}) + "\n")
    print(f"{len(recs)} questions relabelled at tau = {tau:g} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    g = sub.add_parser("get")
    g.add_argument("modality", choices=["video", "audio", "image_text"])
    f = sub.add_parser("forget", help="remove a modality's saved value (its next causal call sweeps again)")
    f.add_argument("modality", choices=["video", "audio", "image_text"])
    p = sub.add_parser("pick")
    p.add_argument("--modality", required=True, choices=["video", "audio", "image_text"])
    p.add_argument("--run", required=True)
    p.add_argument("--tag", required=True, help="tag of the label-only sweep")
    p.add_argument("--dev_frac", type=float, default=0.0,
                   help="fraction of the run used as dev (recorded, so later causal runs skip those questions); "
                        "0 = the dev questions are a separate file")
    p.add_argument("--default", type=float, default=LABEL_TAU)
    p.add_argument("--force", action="store_true")
    r = sub.add_parser("relabel")
    r.add_argument("--run", required=True)
    r.add_argument("--tau", type=float, default=None)
    args = ap.parse_args()
    if args.cmd == "show":
        tab = tau_table()
        for m in ("video", "audio", "image_text"):
            print(f"{m:<11s} " + (f"{tab[m]['tau']:g}  ({tab[m].get('why') or tab[m]['criterion']})" if m in tab
                                  else f"not chosen (fallback {LABEL_TAU:g})"))
    elif args.cmd == "get":
        e = tau_table().get(args.modality)
        if e:
            print(f"{e['tau']:g}")
    elif args.cmd == "forget":
        tab = tau_table()
        if tab.pop(args.modality, None) is not None:
            TAU_FILE.write_text(json.dumps(tab, indent=2) + "\n")
        print(f"{args.modality}: no saved tau")
    elif args.cmd == "pick":
        pick(args)
    else:
        relabel(args)


if __name__ == "__main__":
    main()
