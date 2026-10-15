"""Run OmniTrace v2 on a question file: generate with attention capture, attribute every sentence.

    python scripts/run_attribution.py \
        --questions $OMNITRACE_DATA/question_video.json --model qwen2.5-omni-7b \
        --out runs/nextgqa/qwen2.5-omni-7b --limit 20

    # multi-GPU: one process per GPU (shard i of n), outputs merge by simple concatenation
    CUDA_VISIBLE_DEVICES=0 python scripts/run_attribution.py ... --num_shards 4 --shard_id 0

Outputs in --out:
    configs.jsonl                   one line per launched process (args, library versions, start time)
    records[.shardK].jsonl          one JSON record per question (response, parsed answer, attributions:
                                    per-unit scores for every sentence and for the whole response)
    traces/<id>.npz                 per-token, per-layer, per-unit attention [T, L, U] + generated ids
                                    (+ per-position signal with --dump_positions): enough to recompute any
                                    attention setting on CPU and to replay the response for causal tests
                                    or gradient signals without generating again
Resumable: questions already present in the records file are skipped.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from omnitrace.backends import BACKENDS, get_backend
from omnitrace.evidence import (ATTENTION, attention_attribution, layer_slice, modality_tau, parse_setting,
                                run_modality)
from omnitrace.data import build_doc_prompt, build_mcqa_prompt, load_questions, parse_answer, shuffle_options
from omnitrace.segment import segment_response, token_char_offsets
from omnitrace.timeline import AUDIO, IMAGE, VIDEO
from omnitrace.tracing import SIGNALS, attribute_sentences, signal, token_traces, unit_aggregates


def safe_name(sample_id: str) -> str:
    return sample_id.replace("/", "__")


def run_one(backend, sample, args):
    long_qa = sample.get("task") == "long_qa"
    if long_qa:          # quote-based long-form QA (MMDocRAG): interleaved text + image quotes, no options
        prompt = build_doc_prompt(sample, args.doc_prompt)
    else:
        prompt = build_mcqa_prompt(sample, modality=sample["sources"][0]["modality"], style=getattr(args, "mcqa_prompt", "plain"))
    inputs, tl, meta = backend.prepare(sample, prompt)
    gen = backend.generate(inputs, max_new_tokens=args.max_new_tokens)

    gen_ids, A = gen["gen_ids"], gen["A"]
    content_ids = [i for i in gen_ids if i not in backend.stop_ids]
    response, offsets = token_char_offsets(backend.tokenizer, gen_ids)
    sentences = segment_response(response, offsets)
    pred_idx, parse_mode = (None, "long_qa") if long_qa else parse_answer(response, sample["options"])

    media_modalities = sorted({m for m in (VIDEO, AUDIO, IMAGE) if (tl.pos_modality == m).any()})
    arrays = unit_aggregates(A, tl)
    attribution = {}
    for sig in args.signals:
        if sig == "attention":          # the fixed rule: late layers, attention mass per unit, no argmax
            attribution[sig] = attention_attribution(
                arrays["unit_sum"], sentences, tl.units,
                args.label_tau if args.label_tau is not None else modality_tau(run_modality(tl.units)))
            continue
        S = signal(A, sig)              # per-token argmax + curation (earlier variant, for comparison)
        traces = token_traces(S, tl, media_modalities)
        attribution[sig] = {
            variant: {
                "traces": [[t["unit"], round(t["w"], 6)] for t in tr],
                "sentences": attribute_sentences(sentences, tr, tl),
            }
            for variant, tr in traces.items()
        }

    trace_path = Path(args.out) / "traces" / f"{safe_name(sample['id'])}.npz"
    arrays.update(pos_unit=tl.pos_unit, pos_modality=tl.pos_modality, gen_ids=np.array(gen_ids, np.int64))
    if args.dump_positions in ("signals", "all"):
        # the fixed signal per position (late-layer mean): lets units be re-cut and position-level
        # signals (e.g. gradients) be combined later without a new generation run
        late = layer_slice(parse_setting(ATTENTION)["layers"], A.shape[1])
        arrays["pos_attention"] = A[:, late].mean(1).numpy().astype(np.float16)
        for sig in args.signals:
            if sig != "attention":
                arrays[f"pos_{sig}"] = signal(A, sig).numpy().astype(np.float32)
    if args.dump_positions == "all":
        arrays["pos_layers"] = A.numpy().astype(np.float16)
    np.savez_compressed(trace_path, **arrays)

    return {
        "id": sample["id"],
        "dataset": sample.get("dataset"),
        "model": args.model,
        "prompt": prompt.text,
        "response": response,
        "n_generated": gen["n_generated"],
        "n_content_tokens": len(content_ids),
        "options": sample["options"],                  # as shown in the prompt
        "option_order": sample.get("option_order"),     # shown -> original index (None: not shuffled)
        "pred_idx": pred_idx,
        "parse_mode": parse_mode,
        "answer_idx": sample.get("answer_idx"),
        # long_qa: correctness needs an LLM judge -> None here
        "correct": None if long_qa else ((pred_idx == sample.get("answer_idx")) if pred_idx is not None else False),
        "doc_prompt": args.doc_prompt if long_qa else None,
        "mcqa_prompt": None if long_qa else getattr(args, "mcqa_prompt", "plain"),
        "units": tl.units,
        "sentences": [{"text": s["text"], "token_idx": s["token_idx"], "pos": s["pos"]} for s in sentences],
        "attribution": attribution,
        "meta": {**meta, "gen_seconds": round(gen["gen_seconds"], 2),
                 "n_steps_recorded": gen["n_steps_recorded"], "trace_file": str(trace_path.name)},
    }


def _versions():
    import importlib.metadata as md
    v = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "qwen-omni-utils", "minicpmo-utils", "spacy", "decord", "eva-decord", "torchcodec"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--model", default="qwen2.5-omni-7b", choices=sorted(BACKENDS))
    ap.add_argument("--model_path", default=None, help="local checkpoint dir (default: HF id)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N questions")
    ap.add_argument("--sample", type=int, default=None,
                    help="random N questions (seeded shuffle, first N; nested: sample 100 is a subset of sample 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None, help="only these question ids")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--signals", "--signal", nargs="+", default=["attention"],
                    help="attention = the fixed rule (late-layer attention mass per unit; sentence- and "
                         "response-level scores). attmean / attraw / layerK = v1 argmax + curation (baseline)")
    ap.add_argument("--label_tau", type=float, default=None,
                    help="score -> label rule: a unit is labelled when its score >= tau x the mean of the other "
                         "units of its modality. Default: the sample's modality value in configs/label_tau.json "
                         "(scripts/tau.py); labels can be recomputed offline from the stored scores")
    ap.add_argument("--video_fps", type=float, default=2.0)
    ap.add_argument("--video_total_pixels", type=int, default=20480 * 28 * 28,
                    help="pixel budget over all frames (~20k video tokens max); qwen_omni_utils semantics")
    ap.add_argument("--video_max_pixels", type=int, default=None)
    ap.add_argument("--audio_unit_sec", type=float, default=1.0, help="audio time-unit length (multiple of 0.04 s)")
    ap.add_argument("--audio_in_video", action="store_true",
                    help="video questions with the video's sound track (Qwen2.5-Omni; audio-vs-visual reliance run)")
    ap.add_argument("--image_max_pixels", type=int, default=1024 * 28 * 28,
                    help="per-image pixel cap (<= ~1024 tokens per image); qwen_omni_utils semantics")
    ap.add_argument("--doc_prompt", choices=["plain", "cite"], default="plain",
                    help="long_qa instruction: plain (main) or cite (MMDocRAG citation format; self-citation baseline)")
    ap.add_argument("--mcqa_prompt", choices=["plain", "cite"], default="plain",
                    help="mcqa instruction: plain (main) or cite (asks for cited time spans; self-citation baseline)")
    ap.add_argument("--shuffle_options", action="store_true",
                    help="seeded per-question option order (use for MMAU: answer is A in 117/249); "
                         "pred_idx/answer_idx then refer to the shown order, original order in option_order")
    ap.add_argument("--dump_positions", choices=["none", "signals", "all"], default="none",
                    help="signals: also save the fixed attention signal per position (pos_attention [T, N], fp16; "
                         "plus pos_<signal> of requested v1 signals); all: every layer per position (large)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    suffix = f".shard{args.shard_id}" if args.num_shards > 1 else ""
    rec_path = out / f"records{suffix}.jsonl"
    err_path = out / f"errors{suffix}.jsonl"
    # one line per launched process (shards and phases), so earlier launches are not overwritten
    import datetime
    with (out / "configs.jsonl").open("a") as fc:
        fc.write(json.dumps({**vars(args), "versions": _versions(),
                             "started": datetime.datetime.now().isoformat(timespec="seconds")}) + "\n")

    # resume across ALL record files, so the shard count can change between launches
    done = {json.loads(l)["id"] for f in out.glob("records*.jsonl") for l in f.open() if l.strip()}
    items = load_questions(args.questions, args.limit, ids=args.ids)
    if args.sample:
        order = list(items)
        random.Random(args.seed).shuffle(order)
        items = sorted(order[: args.sample], key=lambda x: x["id"])
    items = items[args.shard_id::args.num_shards]
    todo = [x for x in items if x["id"] not in done]
    if args.shuffle_options:
        todo = [shuffle_options(x, args.seed) for x in todo]
    print(f"{len(items)} questions in shard, {len(done)} done, {len(todo)} to run")

    kw = {"audio_in_video": True} if args.audio_in_video else {}     # only the Qwen2.5-Omni backend takes it
    backend = get_backend(args.model, args.model_path, video_fps=args.video_fps,
                          video_total_pixels=args.video_total_pixels, video_max_pixels=args.video_max_pixels,
                          audio_unit_sec=args.audio_unit_sec, image_max_pixels=args.image_max_pixels, **kw)
    torch.manual_seed(0)
    with rec_path.open("a") as f, err_path.open("a") as ferr:
        bar = tqdm(todo, desc=f"{args.model} shard{args.shard_id}", file=sys.stderr, mininterval=30, ascii=True)
        for i_s, sample in enumerate(bar, 1):
            try:
                rec = run_one(backend, sample, args)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                tqdm.write(f"[{i_s}/{len(todo)}] {sample['id']} n_gen={rec['n_generated']} "
                           f"gen={rec['meta']['gen_seconds']}s", file=sys.stderr)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                ferr.write(json.dumps({"id": sample["id"], "error": "OOM", "msg": str(e)[:500]}) + "\n")
                ferr.flush()
            except Exception as e:
                ferr.write(json.dumps({"id": sample["id"], "error": type(e).__name__,
                                       "msg": str(e)[:500], "tb": traceback.format_exc()[-2000:]}) + "\n")
                ferr.flush()
                if args.limit is not None and args.limit <= 5:
                    raise


if __name__ == "__main__":
    main()
