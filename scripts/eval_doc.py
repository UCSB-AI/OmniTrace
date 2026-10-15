"""Agreement of quote attributions with MMDocRAG gold quotes (CPU; secondary to causal_doc.py).

    python scripts/eval_doc.py --run runs/mmdocrag/qwen2.5-omni-7b \
        --questions $OMNITRACE_DATA/question_image_text.json [--split test] [--correct_ids judged_correct.txt]

Per question, each setting ranks the ~15 quotes by response-level evidence (all response sentences; the gold
set is per question). Metrics vs gold: AUROC, Hit@1, MRR, R@|gold| (recall of the top-|gold| quotes), and the
image share of the top-|gold| quotes vs the gold image share (modality bias). "random" = mean over 200
random rankings of the same question.
Self-citation (P0 baseline): for runs with --doc_prompt cite, the response's [k] / ![..](imagek) citations
are scored with citation precision / recall vs gold.

Data card (§6): free-generation agreement is only meaningful for answers judged correct (LLM judge, e.g.
MMDocRAG's eval_llm_judge.py). Pass the judged-correct ids with --correct_ids (json list or one id per line);
without it all questions are scored and the output says so. The gold set is not minimal (precision is harsh).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from omnitrace.evidence import parse_setting, setting_name
from omnitrace.metrics import auroc
from causal_doc import MASS_MEAN, DocTraceData, doc_split  # noqa: E402

TEXT_CITE = re.compile(r"\[(\d+)\]")
IMAGE_CITE = re.compile(r"\]\((image\d+)\)")


def rank_metrics(scores: np.ndarray, labels: np.ndarray, is_img: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    g = int(labels.sum())
    first = int(np.argmax(labels[order])) if g else 0
    top = order[:g]
    return {"auroc": auroc(scores, labels), "hit1": float(labels[order[0]]), "mrr": 1.0 / (first + 1),
            f"r@gold": float(labels[top].sum() / g) if g else float("nan"),
            "img_share_top": float(is_img[top].mean()) if g else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--settings", nargs="+", default=["mass", MASS_MEAN, "v1", "soft_last"])
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--correct_ids", default=None, help="judged-correct ids (json list or one per line)")
    ap.add_argument("--json", default=None, help="write per-setting means here")
    args = ap.parse_args()

    runp = Path(args.run)
    qs = {q["id"]: q for q in json.loads(Path(args.questions).read_text())}
    keep = doc_split(list(qs), args.split)
    if args.correct_ids:
        txt = Path(args.correct_ids).read_text().strip()
        ok = set(json.loads(txt)) if txt.startswith("[") else {l.strip() for l in txt.splitlines() if l.strip()}
        keep &= ok
    recs = [json.loads(l) for f in sorted(runp.glob("records*.jsonl")) for l in f.open() if l.strip()]
    recs = [r for r in {r["id"]: r for r in recs}.values()
            if r["id"] in keep and (runp / "traces" / r["meta"]["trace_file"]).exists()]
    print(f"{len(recs)} questions (split {args.split}; "
          f"{'judged-correct only' if args.correct_ids else 'ALL answers: not filtered by correctness (see README)'})")

    cfgs = [(spec, parse_setting(spec)) for spec in args.settings]
    acc = {("mass_mean" if s == MASS_MEAN else setting_name(c)): [] for s, c in cfgs}
    acc["random"], cite = [], []
    gold_img_share = []
    rng = np.random.default_rng(0)
    for r in recs:
        q = qs[r["id"]]
        gold = {e["sid"] for e in q["evidence"]}
        td = DocTraceData(np.load(runp / "traces" / r["meta"]["trace_file"]), r)
        sids = [td.units[u]["sid"] for u in td.vids]
        labels = np.array([s in gold for s in sids], float)
        is_img = np.array([td.units[u]["modality"] == "image" for u in td.vids], float)
        if labels.sum() in (0, len(labels)):
            continue
        gold_img_share.append(float(is_img[labels > 0].mean()))
        for spec, cfg in cfgs:
            V = td.evidence(cfg)
            acc["mass_mean" if spec == MASS_MEAN else setting_name(cfg)].append(
                rank_metrics(np.array([V[u] for u in td.vids]), labels, is_img))
        rnd = [rank_metrics(rng.random(len(labels)), labels, is_img) for _ in range(200)]
        acc["random"].append({k: float(np.nanmean([x[k] for x in rnd])) for k in rnd[0]})
        if r.get("doc_prompt") == "cite":
            cited = {f"text{m}" for m in TEXT_CITE.findall(r["response"])} | set(IMAGE_CITE.findall(r["response"]))
            cited &= set(sids)
            cite.append({"precision": len(cited & gold) / len(cited) if cited else 0.0,
                         "recall": len(cited & gold) / len(gold), "n_cited": len(cited), "empty": float(not cited)})

    print(f"gold image share (mean over questions): {np.mean(gold_img_share):.3f}\n")
    keys = ["auroc", "hit1", "mrr", "r@gold", "img_share_top"]
    print(f"{'setting':<34s} " + " ".join(f"{k:>13s}" for k in keys) + "      n")
    out = {}
    for name, rows in acc.items():
        if not rows:
            continue
        m = {k: float(np.nanmean([x[k] for x in rows])) for k in keys}
        out[name] = {**m, "n": len(rows)}
        print(f"{name[:34]:<34s} " + " ".join(f"{100 * m[k]:13.2f}" for k in keys) + f" {len(rows):6d}")
    if cite:
        m = {k: float(np.mean([x[k] for x in cite])) for k in cite[0]}
        out["self_citation"] = {**m, "n": len(cite)}
        print(f"\nself-citation (cite prompt): precision {100 * m['precision']:.2f}  recall {100 * m['recall']:.2f}  "
              f"mean cited {m['n_cited']:.2f}  no citation {100 * m['empty']:.1f}%  (n={len(cite)})")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
