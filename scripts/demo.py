"""Attribute the bundled examples (or your own question file) and print each sentence with its sources.

    python scripts/demo.py                                    # examples/questions.json: one video, one audio, one document question
    python scripts/demo.py --only video --model qwen3-omni-30b-a3b
    python scripts/demo.py --questions my_questions.json --data_root /path/to/media --out demo.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnitrace.api import OmniTrace          # noqa: E402
from omnitrace.backends import BACKENDS      # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def show(sample, out):
    print("=" * 100)
    print("Q:", sample["question"])
    for i, o in enumerate(sample.get("options") or []):
        print(f"   {'ABCDEFGH'[i]}. {o}")
    print("-" * 100)
    for s in out["sentences"]:
        spans = "; ".join(f"{sid} {a:.1f}-{b:.1f}s" for sid, iv in s["spans"].items() for a, b in iv)
        ev = spans or ", ".join(s["sources"]) or "(no source)"
        print(f"{s['text'].strip()[:90]:<92} <- {ev}")
    r = out["response_attribution"]
    print("-" * 100)
    print("whole response <-", "; ".join(f"{sid} {a:.1f}-{b:.1f}s" for sid, iv in r["spans"].items() for a, b in iv)
          or ", ".join(r["sources"]) or "(no source)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-omni-7b", choices=sorted(BACKENDS))
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--questions", default=str(REPO / "examples" / "questions.json"))
    ap.add_argument("--data_root", default=None, help="root of relative media paths (default: the question file's folder)")
    ap.add_argument("--only", choices=["video", "audio", "image_text"], default=None)
    ap.add_argument("--out", default=None, help="write the full outputs (scores per unit, labels) as JSON")
    args = ap.parse_args()

    samples = json.loads(Path(args.questions).read_text())
    if args.only:
        samples = [s for s in samples if s.get("modality") == args.only]
    root = args.data_root or str(Path(args.questions).resolve().parent)
    ot = OmniTrace(args.model, args.model_path)
    results = []
    for sample in samples:
        out = ot.trace(sample, data_root=root)
        show(sample, out)
        results.append({"id": sample.get("id"), **out})
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
