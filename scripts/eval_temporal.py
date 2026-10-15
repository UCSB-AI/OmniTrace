"""Evaluate temporal attribution of a run against human evidence spans (CPU only).

    python scripts/eval_temporal.py --run runs/nextgqa/qwen2.5-omni-7b \
        --questions $OMNITRACE_DATA/question_video.json [--json out.json]

Per (signal x trace variant x sentence aggregation) it reports, over questions with evidence:
  set metrics on the union of predicted spans vs union of GT spans:
      Time-P (= IoP), Time-R, Time-F1, IoU, coverage (predicted fraction of video), empty rate
  NExT-GQA official-style metrics on the hull [min start, max end] of the prediction, max over GT
  segments (code/TempGQA/eval_ground.py): mIoP, IoP@0.3, IoP@0.5, mIoU, IoU@0.3, IoU@0.5, Acc&GQA
And, threshold-free from traces/*.npz (per-unit attention mass summed over generated tokens):
      AUROC (unit = positive if >=50% inside GT), Hit@1 (top unit overlaps GT)
Baselines: random units matched to each prediction's size (5 seeds), full video.
Aggregations: "all" = union over all response sentences; "answer" = sentences containing the final
answer (fallback: last sentence).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnitrace.timeline import merge_intervals  # noqa: E402

from omnitrace.metrics import (  # noqa: E402
    Span, _inter, _len, answer_sentence_ids, auroc, mean, official_metrics, question_gt, set_metrics,
    unit_overlap_labels,
)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--json", default=None, help="write full results here")
    ap.add_argument("--no_traces", action="store_true", help="skip npz-based AUROC/Hit@1")
    ap.add_argument("--correct_only", action="store_true",
                    help="agreement only on correctly answered questions (use with the answer rows)")
    args = ap.parse_args()

    run = Path(args.run)
    qs = {q["id"]: q for q in json.loads(Path(args.questions).read_text())}
    recs = [json.loads(l) for f in sorted(run.glob("records*.jsonl")) for l in f.open() if l.strip()]
    recs = list({r["id"]: r for r in recs}.values())
    print(f"{len(recs)} records")

    acc = [r["correct"] for r in recs]
    parsed = [r["pred_idx"] is not None for r in recs]
    groups = defaultdict(list)
    rows = defaultdict(lambda: defaultdict(list))   # key -> metric -> values
    rng = random.Random(0)

    for r in recs:
        q = qs[r["id"]]
        if not q.get("evidence") or (args.correct_only and not r["correct"]):
            continue
        ev = q["evidence"][0]
        sid = ev["sid"]
        _, gt, gt_segments, dur = question_gt(q)          # clipped to [0, duration]
        units = r["units"]
        vids, labels = unit_overlap_labels(units, sid, gt)
        qtype = q.get("meta", {}).get("qtype_group", "all")
        groups[qtype].append(r["id"])
        ans_ids = answer_sentence_ids(r["sentences"], q["options"], r["pred_idx"])

        # span labels exist only for the v1 signals; the fixed `attention` output is scores
        legacy = {k: v for k, v in r["attribution"].items() if k != "attention"}
        for sig, variants in legacy.items():
            for var, att in variants.items():
                sents = att["sentences"]
                for agg, idx in (("all", range(len(sents))), ("answer", ans_ids)):
                    pred = merge_intervals([tuple(x) for i in idx if i < len(sents)
                                            for x in sents[i]["spans"].get(sid, [])])
                    pred = [tuple(x) for x in pred]
                    key = f"{sig}/{var}/{agg}"
                    m = {**set_metrics(pred, gt), **official_metrics(pred, gt_segments)}
                    m["coverage"] = _len(pred) / dur if dur else 0.0
                    m["empty"] = float(not pred)
                    m["acc_gqa"] = float(r["correct"] and m["iop"] >= 0.5)
                    for k, v in m.items():
                        rows[key][k].append(v)
                    # size-matched random baseline: same number of units, uniformly at random
                    n_sel = len({u for i in idx if i < len(sents) for u in sents[i]["units"]
                                 if units[u]["sid"] == sid})
                    if vids and n_sel:
                        for seed in range(5):
                            pick = rng.sample(vids, min(n_sel, len(vids)))
                            rp = [tuple(x) for x in merge_intervals([(units[u]["t0"], units[u]["t1"]) for u in pick])]
                            rm = {**set_metrics(rp, gt), **official_metrics(rp, gt_segments)}
                            for k, v in rm.items():
                                rows[f"random~{key}"][k].append(v)
        full = [(0.0, dur)]
        for k, v in {**set_metrics(full, gt), **official_metrics(full, gt_segments)}.items():
            rows["baseline/full_video"][k].append(v)

        # threshold-free scores from the npz traces
        npz = run / "traces" / r["meta"]["trace_file"]
        if not args.no_traces and npz.exists() and vids and labels.any() and not labels.all():
            z = np.load(npz)
            usum = z["unit_sum"]                        # [T, L, U]
            for sig, layer_sel in (("attmean", slice(None)), ("attraw", slice(-1, None))):
                score = usum[:, layer_sel, :].mean(1).sum(0)[vids]
                rows[f"{sig}/mass"]["auroc"].append(auroc(score, labels))
                rows[f"{sig}/mass"]["hit1"].append(float(labels[int(np.argmax(score))]))
            for sig in legacy:
                tr = legacy[sig].get("video", {}).get("traces", [])
                votes = np.zeros(len(units))
                for u, w in tr:
                    if u >= 0:
                        votes[u] += w
                sc = votes[vids]
                rows[f"{sig}/video_votes"]["auroc"].append(auroc(sc, labels))
                rows[f"{sig}/video_votes"]["hit1"].append(float(labels[int(np.argmax(sc))]) if sc.any() else 0.0)
            rows["random/scores"]["auroc"].append(0.5)
            rows["random/scores"]["hit1"].append(float(labels.mean()))

    summary = {
        "n_records": len(recs),
        "accuracy": mean(acc),
        "parse_rate": mean(parsed),
        "by_group": {g: {"n": len(ids), "accuracy": mean([next(x for x in recs if x["id"] == i)["correct"] for i in ids])}
                     for g, ids in groups.items()},
        "attribution": {k: {m: mean(v) for m, v in d.items()} | {"n": len(next(iter(d.values())))}
                        for k, d in rows.items()},
    }
    print(f"accuracy {summary['accuracy']:.4f}  parse rate {summary['parse_rate']:.4f}")
    cols = ["time_p", "time_r", "time_f1", "iou", "iop", "iou_hull", "coverage", "empty", "acc_gqa"]
    print(f"{'setting':42s} " + " ".join(f"{c:>8s}" for c in cols))
    for k in sorted(summary["attribution"]):
        d = summary["attribution"][k]
        if "time_f1" in d:
            print(f"{k:42s} " + " ".join(f"{100 * d.get(c, float('nan')):8.2f}" for c in cols))
    print(f"\n{'threshold-free':42s} {'AUROC':>8s} {'Hit@1':>8s} {'n':>6s}")
    for k in sorted(summary["attribution"]):
        d = summary["attribution"][k]
        if "auroc" in d:
            print(f"{k:42s} {100 * d['auroc']:8.2f} {100 * d['hit1']:8.2f} {d['n']:6d}")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
