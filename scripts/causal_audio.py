"""Causal faithfulness test for audio attribution (GPU), plus CPU summary. Audio twin of causal_video.py.

Question: if we hide the audio seconds an attribution setting points to, does the model's confidence in
its *own* text drop more than when we hide the same amount of random audio?

    # GPU, one process per shard, resumable across shard counts
    CUDA_VISIBLE_DEVICES=0 python -u scripts/causal_audio.py run --run runs/mmau/qwen2.5-omni-7b \
        --questions $OMNITRACE_DATA/question_audio.json --tag v1 --num_shards 2 --shard_id 0
    # CPU summary (same tables as causal_video.py); --min_units 5 drops clips with <= 4 one-second units
    python scripts/causal_audio.py summarize --run runs/mmau/qwen2.5-omni-7b --tag v1 [--subset correct] [--min_units 5]

Hiding audio (--fill blank, default): the log-mel frames of the chosen time units (1 unit = 25 tokens = 1 s
by default) are set to silence before the audio encoder (an empty placeholder; omnitrace/ablation.py).
Masking-choice ablations: --fill mean / zero replace the audio encoder's output embeddings of those units by
the clip's mean embedding / zeros. Token positions, prompt and TMRoPE time ids are unchanged either way.
The model's original response (saved gen_ids) is teacher-forced; we read the per-token log-probs.

Variants, targets and the summary are those of causal_video.py (del_top / del_bottom / keep_top vs random
sets of the same size, response and answer-letter log-probs, per-sentence own/cross drops). No labels are
needed, so every question of the run counts (MMAU has no evidence yet). A "gt" variant is added only for
questions whose record in --questions carries audio evidence spans (after the human annotation is imported).
Prompts are rebuilt exactly as in the run, including the seeded option order of --shuffle_options.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
import zlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from omnitrace.evidence import tag_files  # noqa: E402
from omnitrace.external import attach, ext_names, load_external, probe_variants, set_rows
from omnitrace.evidence import (TraceData, label_keys, parse_setting, predicted_units, rank_frames, setting_name, split_ids,
                                take_budget)
from omnitrace.metrics import unit_overlap_labels
from omnitrace.timeline import merge_intervals
# shared with the video test (same variant names, so the same summary applies)
from causal_video import CONTENT_POS, answer_token_index, char_offsets, summarize  # noqa: E402

AUDIO_BATCH_KEYS = ("input_features", "feature_attention_mask")


class AudioTraceData(TraceData):
    """TraceData whose candidate time units are the audio units (evidence.py's `is_video`/`vids` are the
    temporal-media units of the question; scope=video therefore means 'audio units only' here)."""

    def __init__(self, arrays, rec, options):
        super().__init__(arrays, rec, options)
        self.is_video = np.array([u["modality"] == "audio" for u in self.units])
        self.vids = [u["uid"] for u in self.units if u["modality"] == "audio"]


class AudioAblation:
    """Forward hook on the audio encoder. `masks[b]` = audio-token rows (within one clip) to overwrite in
    batch copy b; the encoder output holds len(masks) copies of the clip back to back."""

    def __init__(self, audio_tower, fill: str = "mean"):
        self.masks = None
        self.fill = fill
        self.handle = audio_tower.register_forward_hook(self._hook)

    def _hook(self, module, inputs, out):
        if self.masks is None:
            return None
        import torch
        t = out if torch.is_tensor(out) else getattr(out, "last_hidden_state", None)
        if t is None or not torch.is_tensor(t):
            raise RuntimeError("cannot find audio embeddings in the audio encoder output")
        n = t.shape[0] // len(self.masks)
        new = t.clone()
        for b, rows in enumerate(self.masks):
            if rows is not None and len(rows):
                block = t[b * n:(b + 1) * n]
                new[b * n + rows] = block.mean(0, keepdim=True) if self.fill == "mean" else 0
        if torch.is_tensor(out):
            return new
        out.last_hidden_state = new
        return out


def batch_inputs(inputs, ids, B):
    """Repeat the (prompt + response) inputs B times along the batch axis (one clip per copy)."""
    import torch
    seq = torch.cat([inputs["input_ids"], ids], 1)
    full = {"input_ids": seq.repeat(B, 1)}
    full["attention_mask"] = torch.ones_like(full["input_ids"])
    for k, v in inputs.items():
        if k in ("input_ids", "attention_mask"):
            continue
        full[k] = v.repeat(B, *([1] * (v.dim() - 1))) if k in AUDIO_BATCH_KEYS and torch.is_tensor(v) else v
    return full


def audio_gt(q, sid_units):
    """Merged GT spans on the audio source, or None when the question has no audio evidence yet."""
    for ev in q.get("evidence") or []:
        if ev.get("spans") and ev["sid"] == sid_units:
            dur = float(next(s for s in q["sources"] if s["sid"] == ev["sid"])["duration"])
            return [(max(0.0, a), min(dur, b)) for a, b in merge_intervals(ev["spans"]) if min(dur, b) > max(0.0, a)]
    return None


def causal_question(backend, ablate, q, rec, z, args):
    """All hidden-unit variants of one question -> result row (same layout as causal_video.py)."""
    import torch
    from omnitrace.data import LETTERS, build_mcqa_prompt, shuffle_options

    thinker, tok = getattr(backend, "thinker", None), backend.tokenizer
    if rec.get("option_order") is not None:
        q = shuffle_options(q, args.option_seed)
        if q["option_order"] != rec["option_order"]:
            raise RuntimeError("option order differs from the run (wrong --option_seed?)")
    prompt = build_mcqa_prompt(q, modality="audio", style=rec.get("mcqa_prompt") or "plain")
    inputs, tl, meta = backend.prepare(q, prompt)
    n_prompt = int(inputs["input_ids"].shape[1])
    if n_prompt != rec["meta"]["n_prompt_tokens"] or len(tl.units) != len(rec["units"]):
        raise RuntimeError(f"input mismatch: {n_prompt} vs {rec['meta']['n_prompt_tokens']}")
    gen_ids = [int(x) for x in z["gen_ids"]]
    T = len(gen_ids)
    letters = LETTERS[: len(q["options"])]
    letter_ids = [tok.convert_tokens_to_ids(l) for l in letters]
    text, offs = char_offsets(tok, gen_ids)
    ans_idx = answer_token_index(text, offs, letters)

    td = AudioTraceData(z, rec, q["options"])
    attach(td, args.ext_store, rec["id"])
    src = next(s for s in q["sources"] if s["modality"] == "audio")
    dur = float(src["duration"])
    k_unit, n_audio = meta.get("audio_unit_tokens"), meta["n_audio_tokens"]   # k_unit: Qwen2.5 path only (rows_of)
    gidx = {u["uid"]: int(u["label"][1:]) for u in tl.units if u["modality"] == "audio"}
    all_a = list(gidx)
    dev = inputs["input_ids"].device

    def rows_of(units):
        if not units:
            return None
        return torch.tensor([r for u in units for r in range(gidx[u] * k_unit, min((gidx[u] + 1) * k_unit, n_audio))],
                            dtype=torch.long, device=dev)

    # content tokens per sentence; content sentences = >=3 content words, not the "Answer: X" line
    sent_tok = defaultdict(list)
    for k, si in td.tok_sent.items():
        if k < T and gen_ids[k] not in backend.stop_ids:
            sent_tok[si].append(k)
    content_sents = [si for si, ks in sorted(sent_tok.items())
                     if sum(td.tok_pos[k] in CONTENT_POS for k in ks) >= 3
                     and not rec["sentences"][si]["text"].lower().startswith("answer")]
    resp_tok = sorted(k for ks in sent_tok.values() for k in ks)

    variants = {"orig": ()}
    rng = random.Random(zlib.crc32(rec["id"].encode()))
    rand_sets = {}

    def rand_k(k):
        if k not in rand_sets:
            rand_sets[k] = [tuple(sorted(rng.sample(all_a, k))) for _ in range(args.rand_seeds)]
            for i, s in enumerate(rand_sets[k]):
                variants[f"rand|{k}|{i}"] = s
                variants[f"keeprand|{k}|{i}"] = tuple(u for u in all_a if u not in set(s))

    plan = {}
    for cfg in [parse_setting(s) for s in args.settings]:
        ranked, _ = rank_frames(td, cfg)
        name = setting_name(cfg)
        for b in args.budgets:
            top = take_budget(ranked, td.units, dur, b)
            k = len(top)
            variants[f"{name}|{b}|del_top"] = tuple(sorted(top))
            variants[f"{name}|{b}|del_bottom"] = tuple(sorted(ranked[-k:]))
            variants[f"{name}|{b}|keep_top"] = tuple(u for u in all_a if u not in set(top))
            rand_k(k)
            plan[f"{name}|{b}"] = k
    gt = audio_gt(q, src["sid"])
    if gt and not args.labels_only:
        uids, labels = unit_overlap_labels(td.units, src["sid"], gt, short_spans=True)
        gt_units = [u for u, l in zip(uids, labels) if l]
        if gt_units:
            variants["gt|del_top"] = tuple(sorted(gt_units))
            rand_k(len(gt_units))
            plan["gt"] = len(gt_units)
    sent_plan = {}
    for cfg in [parse_setting(s) for s in args.sent_settings]:
        name = setting_name(cfg)
        for j in content_sents:
            ranked, _ = rank_frames(td, {**cfg, "sents": {j}})
            top = take_budget(ranked, td.units, dur, args.sent_budget)
            variants[f"sent|{name}|{j}"] = tuple(sorted(top))
            rand_k(len(top))
            sent_plan[f"{name}|{j}"] = len(top)

    # ---- the labels the fixed setting outputs (score >= tau x mean of the other units): hide exactly those
    label_sizes = {}
    for lkey, tau in ([] if args.probe or getattr(args, "gt_only", False) else label_keys(args.label_tau, args.label_taus, "audio")):
        lab = predicted_units(td, tau)
        label_sizes[lkey] = {"tau": tau, "response": len(lab), "sent": {}}
        if lab and len(lab) < len(all_a):
            variants[f"{lkey}|del_top"] = tuple(sorted(lab))
            variants[f"{lkey}|keep_top"] = tuple(u for u in all_a if u not in set(lab))
            rand_k(len(lab))
            plan[lkey] = len(lab)
        for j in ([] if args.no_sent else content_sents):
            lab = predicted_units(td, tau, {j})
            label_sizes[lkey]["sent"][str(j)] = len(lab)
            if lab and len(lab) < len(all_a):
                variants[f"sent|{lkey}|{j}"] = tuple(sorted(lab))
                rand_k(len(lab))
                sent_plan[f"{lkey}|{j}"] = len(lab)

    from omnitrace.ablation import blank_audio
    ext_sizes = set_rows(td, args.ext_sets, all_a, variants, plan, sent_plan, rand_k, content_sents,
                         args.settings[:1], rank_frames, parse_setting, setting_name)
    probe = probe_variants(args.probe, all_a, rec["id"], args.probe_n) if args.probe else {}
    variants.update(probe)

    uniq = sorted(set(variants.values()))
    ids = torch.tensor([gen_ids], device=dev)
    outs = []
    with torch.no_grad():
        for i in range(0, len(uniq), args.batch):
            chunk = [list(u) for u in uniq[i:i + args.batch]]
            if ablate is None:            # the backend hides units itself (MiniCPM-o: embedding level)
                res = L = None
                logits = backend.hidden_logits(inputs, gen_ids, chunk, args.fill)
            else:
                full = batch_inputs(inputs, ids, len(chunk))
                if args.fill == "blank":
                    blank_audio(full["input_features"], [(rows_of(u).tolist() if u else None) for u in chunk])
                else:
                    ablate.masks = [rows_of(u) for u in chunk]
                try:
                    try:
                        res = thinker(**full, use_audio_in_video=False, use_cache=False, logits_to_keep=T + 1)
                    except TypeError:
                        res = thinker(**full, use_audio_in_video=False, use_cache=False)
                finally:
                    ablate.masks = None
                L = res.logits
                logits = L[:, :-1] if L.shape[1] == T + 1 else L[:, n_prompt - 1: n_prompt - 1 + T]
            lp = torch.log_softmax(logits.float(), -1)
            tgt = torch.tensor(gen_ids, device=lp.device)
            tok_lp = lp.gather(-1, tgt.view(1, T, 1).expand(lp.shape[0], T, 1))[..., 0]
            am = lp.argmax(-1)
            for b in range(lp.shape[0]):
                ap = int(lp[b, ans_idx, letter_ids].argmax()) if ans_idx is not None else None
                outs.append((tok_lp[b].cpu().numpy(), am[b].cpu().numpy(), ap))
            del res, L, logits, lp
    by_set = dict(zip(uniq, outs))

    def summ(lp, ap):
        d = {"resp_lp": float(lp[resp_tok].mean()) if resp_tok else float("nan"),
             "sent_lp": {str(si): float(lp[ks].mean()) for si, ks in sent_tok.items()}}
        if ans_idx is not None:
            d["ans_lp"] = float(lp[ans_idx])
            d["ans_pred"] = ap
        return d

    agree = float((by_set[()][1] == np.array(gen_ids)).mean())
    return {
        "id": rec["id"], "n_units": len(all_a), "duration": dur, "correct": rec["correct"],
        "task_domain": q.get("meta", {}).get("task_domain"), "sub_category": q.get("meta", {}).get("sub_category"),
        "fill": args.fill, "label_tau": {k: v["tau"] for k, v in label_sizes.items()}, "label_sizes": label_sizes,
        "argmax_agree": agree, "content_sents": [str(j) for j in content_sents],
        "plan": plan, "sent_plan": sent_plan, "n_forward": len(uniq),
        "ext_sizes": ext_sizes, "probe": args.probe, "probe_sets": {k: list(v) for k, v in probe.items()},
        "sent_ntok": {str(si): len(ks) for si, ks in sent_tok.items()},
        "results": {name: summ(by_set[s][0], by_set[s][2]) for name, s in variants.items()},
    }


def run(args):
    import torch
    from tqdm import tqdm
    from omnitrace.backends import get_backend

    for k, v in (("ext_sets", []), ("probe", None), ("probe_n", 32)):   # callers without the baseline args
        setattr(args, k, getattr(args, k, v))
    runp = Path(args.run)
    out_dir = runp / "causal"
    out_dir.mkdir(exist_ok=True)
    suffix = f".shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = out_dir / f"causal_{args.tag}{suffix}.jsonl"
    done = {json.loads(l)["id"] for f in tag_files(out_dir, args.tag) for l in f.open() if l.strip()}

    run_cfg = json.loads((runp / "configs.jsonl").read_text().splitlines()[0])
    if args.option_seed is None:
        args.option_seed = run_cfg.get("seed", 0)
    qs = {q["id"]: q for q in json.loads(Path(args.questions).read_text())}
    recs = [json.loads(l) for f in sorted(runp.glob("records*.jsonl")) for l in f.open() if l.strip()]
    recs = sorted({r["id"]: r for r in recs}.values(), key=lambda r: r["id"])
    recs = [r for r in recs if r["id"] in qs and (runp / "traces" / r["meta"]["trace_file"]).exists()]
    if args.n:
        random.Random(args.seed).shuffle(recs)
        recs = sorted(recs[: args.n], key=lambda r: r["id"])
    keep = split_ids([r["id"] for r in recs], "audio", args.split, args.dev_frac)
    recs = [r for r in recs if r["id"] in keep]
    if getattr(args, "gt_only", False):          # only questions with human spans
        recs = [r for r in recs if qs[r["id"]].get("evidence")]
    recs = [r for r in recs[args.shard_id::args.num_shards] if r["id"] not in done]
    print(f"{len(recs)} questions to run (split {args.split}, shard {args.shard_id}/{args.num_shards})", file=sys.stderr, flush=True)

    backend = get_backend(run_cfg["model"], args.model_path or run_cfg.get("model_path"),
                          audio_unit_sec=run_cfg.get("audio_unit_sec", 1.0))
    args.ext_store = load_external(runp, ext_names(args.settings + args.sent_settings) + list(args.ext_sets))
    ablate = None if hasattr(backend, "hidden_logits") else \
        AudioAblation(backend.thinker.audio_tower, "zero" if args.fill == "zero" else "mean")
    t_start = time.time()
    with out_path.open("a") as fout:
        bar = tqdm(recs, desc=f"causal-audio {args.tag} shard{args.shard_id}", file=sys.stderr, mininterval=30, ascii=True)
        for i_rec, rec in enumerate(bar, 1):
            try:
                z = np.load(runp / "traces" / rec["meta"]["trace_file"])
                row = causal_question(backend, ablate, qs[rec["id"]], rec, z, args)
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                el = time.time() - t_start
                tqdm.write(f"[{i_rec}/{len(recs)}] {rec['id']}: {row['n_forward']} variants, argmax_agree "
                           f"{row['argmax_agree']:.3f} | elapsed {el / 60:.1f} min, "
                           f"ETA {el / i_rec * (len(recs) - i_rec) / 60:.1f} min", file=sys.stderr)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                tqdm.write(f"[OOM] {rec['id']} (try a smaller --batch)", file=sys.stderr)
            except Exception:
                tqdm.write(f"[error] {rec['id']}\n{traceback.format_exc()[-1500:]}", file=sys.stderr)


def summarize_audio(args):
    """causal_video.summarize, optionally on clips with >= --min_units audio units only (41/249 MMAU clips
    have <= 4 units, where every budget collapses to 1-2 units). The filtered rows are summarized from a
    temporary copy; the CSV is written next to the full one as summary_<tag>_min<k>_<subset>.csv."""
    args.modality = "audio"
    if not args.min_units:
        return summarize(args)
    import shutil
    import tempfile
    runp = Path(args.run)
    rows = [l for f in tag_files(runp / "causal", args.tag) for l in f.open()
            if l.strip() and json.loads(l).get("n_units", 0) >= args.min_units]
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "causal").mkdir()
        (Path(tmp) / "causal" / f"causal_{args.tag}.jsonl").write_text("".join(rows))
        print(f"clips with >= {args.min_units} audio units only")
        summarize(argparse.Namespace(run=tmp, tag=args.tag, subset=args.subset, split=args.split, modality="audio", no_labels=args.no_labels,
                                     dev_frac=args.dev_frac))
        sp = "" if args.split == "all" else f"_{args.split}"
        shutil.copy(Path(tmp) / "causal" / f"summary_{args.tag}{sp}_{args.subset}.csv",
                    runp / "causal" / f"summary_{args.tag}_min{args.min_units}{sp}_{args.subset}.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True)
    r.add_argument("--questions", required=True)
    r.add_argument("--model_path", default=None)
    r.add_argument("--n", type=int, default=None, help="seeded random subset of the run (default: all)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--option_seed", type=int, default=None, help="--seed of the run (for --shuffle_options; default: read from the run config)")
    r.add_argument("--settings", nargs="+", default=["mass", "v1", "soft_last"])
    r.add_argument("--budgets", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    r.add_argument("--sent_settings", nargs="+", default=["mass", "v1"])
    r.add_argument("--sent_budget", type=float, default=0.2)
    r.add_argument("--rand_seeds", type=int, default=3)
    r.add_argument("--fill", choices=["blank", "mean", "zero"], default="blank",
                   help="blank (default): silence in the log-mel input (empty placeholder); mean / zero: replace the "
                        "audio-encoder output embeddings (masking-choice ablations)")
    r.add_argument("--label_tau", type=float, default=None,
                   help="score -> label rule constant (default: the modality's value in configs/label_tau.json)")
    r.add_argument("--label_taus", nargs="+", type=float, default=None,
                   help="sweep: label rows for each of these values (keys label@<tau>) instead of the saved one")
    r.add_argument("--labels_only", action="store_true", help="only the label rows (no budget / human-label rows)")
    r.add_argument("--gt_only", action="store_true",
                   help="only the human-evidence rows (hide the annotated seconds vs random), on questions with spans")
    r.add_argument("--no_sent", action="store_true", help="no sentence-level label rows (cheap tau sweep)")
    r.add_argument("--split", choices=["auto", "all", "dev", "test"], default="auto",
                   help="dev = seeded fraction for choosing tau, test = the rest; auto = test when this modality's "
                        "tau was chosen on such a dev split (configs/label_tau.json), else all")
    r.add_argument("--dev_frac", type=float, default=None)
    r.add_argument("--ext_sets", nargs="*", default=[], help="set-type baselines (<run>/baselines/<name>.jsonl), e.g. selfcite")
    r.add_argument("--probe", choices=["loo", "contextcite"], default=None,
                   help="only the ablation forwards a perturbation baseline is fitted on")
    r.add_argument("--probe_n", type=int, default=32, help="contextcite: random ablations per question")
    r.add_argument("--batch", type=int, default=8, help="hidden-unit variants per forward pass")
    r.add_argument("--tag", default="v1", help="output name; use a new tag for a new configuration")
    r.add_argument("--num_shards", type=int, default=1)
    r.add_argument("--shard_id", type=int, default=0)
    s = sub.add_parser("summarize")
    s.add_argument("--run", required=True)
    s.add_argument("--tag", default="v1")
    s.add_argument("--subset", choices=["all", "correct", "wrong"], default="all")
    s.add_argument("--no_labels", action="store_true", help="leave out the label rows (e.g. rows made at an old tau)")
    s.add_argument("--min_units", type=int, default=0, help="only clips with at least this many audio units")
    s.add_argument("--split", choices=["all", "dev", "test"], default="all", help="dev / test of the tau selection")
    s.add_argument("--dev_frac", type=float, default=None)
    args = ap.parse_args()
    if args.cmd == "run" and args.gt_only:
        args.settings, args.sent_settings = [], []
    if args.cmd == "run" and (args.labels_only or args.probe):
        args.settings, args.sent_settings = [], []
        args.labels_only = True
    run(args) if args.cmd == "run" else summarize_audio(args)


if __name__ == "__main__":
    main()
