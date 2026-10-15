"""How the score -> label rule behaves on a finished run, for several values of tau (CPU).

    python scripts/label_stats.py --run runs/nextgqa/qwen2.5-omni-7b --questions $OMNITRACE_DATA/question_video.json \
        --taus 1.5 2 3 4

Labels are recomputed from the saved traces with the fixed attention setting (omnitrace/evidence.py:
attention_attribution + select_units), so no GPU and no new generation run is needed, and runs made before
labels were written to the records work too. Per tau:
  response level   labelled units per question, their share of the candidate units, questions with no label
  sentence level   labelled units per sentence, sentences that abstain (no label)
  vs human labels  (questions with human evidence) precision / recall of the
                   response-level label: time overlap for video/audio spans, set overlap for quotes.
                   Chance precision = the share of the source that is evidence.
                   f1 = mean per-question F1 (empty label = 0): the criterion for choosing tau.
The saved values (configs/label_tau.json) are 1.5 for every modality and model.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from omnitrace.evidence import attention_attribution, candidate_groups
from omnitrace.metrics import _len, question_gt, set_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--questions", default=None, help="question file with evidence (for the human-label columns)")
    ap.add_argument("--taus", nargs="+", type=float, default=[1.5, 2.0, 3.0, 4.0])
    ap.add_argument("--correct_only", action="store_true", help="human-label columns on correctly answered questions only")
    args = ap.parse_args()

    runp = Path(args.run)
    qs = {q["id"]: q for q in json.loads(Path(args.questions).read_text())} if args.questions else {}
    recs = [json.loads(l) for f in sorted(runp.glob("records*.jsonl")) for l in f.open() if l.strip()]
    recs = [r for r in {r["id"]: r for r in recs}.values() if (runp / "traces" / r["meta"]["trace_file"]).exists()]
    acc = {t: defaultdict(list) for t in args.taus}
    for r in recs:
        z = np.load(runp / "traces" / r["meta"]["trace_file"])
        units = r["units"]
        n_cand = sum(len(g) for g in candidate_groups(units).values())
        q = qs.get(r["id"])
        ev = (q or {}).get("evidence")
        use_gt = bool(ev) and (not args.correct_only or r.get("correct"))
        for t in args.taus:
            att = attention_attribution(z["unit_sum"], r["sentences"], units, t)
            a, resp = acc[t], att["response"]
            a["resp_units"].append(len(resp["units"]))
            a["resp_share"].append(len(resp["units"]) / max(n_cand, 1))
            a["resp_empty"].append(float(not resp["units"]))
            for s in att["sentences"]:
                a["sent_units"].append(len(s["units"]))
                a["sent_abstain"].append(float(not s["units"]))
            if not use_gt:
                continue
            if ev[0].get("spans"):                      # temporal evidence: time overlap
                sid, gt, _, dur = question_gt(q)
                pred = [tuple(x) for x in resp["spans"].get(sid, [])]
                m = set_metrics(pred, gt)
                a["chance_p"].append(_len(gt) / dur if dur else float("nan"))
            else:                                       # quote evidence: set overlap
                gold = {e["sid"] for e in ev}
                pred = set(resp["sources"])
                pool = [units[u]["sid"] for g in candidate_groups(units).values() for u in g]
                m = {"time_p": len(pred & gold) / len(pred) if pred else 0.0, "time_r": len(pred & gold) / len(gold)}
                a["chance_p"].append(len(gold & set(pool)) / max(len(pool), 1))
            if pred:
                a["precision"].append(m["time_p"])
            a["recall"].append(m["time_r"])
            pr = m["time_p"] + m["time_r"]              # per-question F1; an empty label scores 0
            a["f1"].append(2 * m["time_p"] * m["time_r"] / pr if pred and pr else 0.0)

    cols = ["resp_units", "resp_share", "resp_empty", "sent_units", "sent_abstain", "precision", "chance_p", "recall", "f1"]
    print(f"{len(recs)} questions; human-label columns on {len(acc[args.taus[0]]['recall'])} "
          f"({'correct answers only' if args.correct_only else 'all answers'}); precision over non-empty labels")
    print(f"{'tau':>6s} " + " ".join(f"{c:>12s}" for c in cols))
    for t in args.taus:
        a = acc[t]
        print(f"{t:6.2f} " + " ".join(f"{np.mean(a[c]):12.3f}" if a[c] else f"{'-':>12s}" for c in cols))


if __name__ == "__main__":
    main()
