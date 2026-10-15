"""Causal faithfulness test for quote attribution on image+text QA (MMDocRAG; GPU), plus CPU summary.
Quote twin of causal_video.py / causal_audio.py.

Question: if we hide the quotes an attribution setting ranks highest, does the model's confidence in its
*own* long answer drop more than when we hide the same number of random quotes?

    # GPU, one process per shard, resumable across shard counts
    CUDA_VISIBLE_DEVICES=0 python -u scripts/causal_doc.py run --run runs/mmdocrag/qwen2.5-omni-7b \
        --questions $OMNITRACE_DATA/question_image_text_dev.json --tag v1_dev --num_shards 2 --shard_id 0
    # CPU summary (same tables as causal_video.py)
    python scripts/causal_doc.py summarize --run runs/mmdocrag/qwen2.5-omni-7b \
        --questions $OMNITRACE_DATA/question_image_text_dev.json --tag v1_dev

Units are whole quotes (~10 text passages + ~5 images per question). Hiding a quote keeps its positions and
the prompt layout. --fill blank (default; an empty placeholder of the same length, omnitrace/ablation.py):
  image quote -> its pixels are set to black before the vision encoder;
  text quote  -> its tokens are replaced by space tokens.
--fill mean (masking-choice ablation; the older variant):
  image quote -> its vision-encoder output rows are replaced by the mean image embedding of the question
                 (all its images), via a forward hook on thinker.visual;
  text quote  -> its token input embeddings are replaced by the mean input embedding of all text-quote
                 tokens of the question, via a forward hook on the input embedding layer.
The model's original response (saved gen_ids) is teacher-forced; we read per-token log-probs.

Budgets are counts of hidden quotes (--ks, default 1 2 3), not fractions of time. Random references hide the
same number of random quotes. Quotes differ a lot in size (an image is hundreds of tokens, a text quote
tens), so every row also stores the tokens each variant hides (`hidden_tokens`).
Variants: del_top / del_bottom / keep_top (keep only the top-k quotes) / rand / keeprand, "gt" = hide the
gold quotes (human labels) vs random of the same count, and per-sentence own/cross drops.

Ablation: `unit=sum` (preset mass) vs `unit=mean` (mass per token,
MASS_MEAN below); the sum is the default. Only questions in --questions are used. --split dev/test re-splits
whatever file is given (seeded 500 / rest).
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
from omnitrace.evidence import TraceData, label_keys, parse_setting, predicted_units, rank_frames, setting_name
# shared with the video test (same variant names, so the same summary applies)
from causal_video import CONTENT_POS, summarize  # noqa: E402

MASS_MEAN = "layers=late,unit=mean,contrast=none,scope=video,vote=mass,pos=off,gamma=0,sents=all"
DEV_N, DEV_SEED = 500, 0
IMAGE_BATCH_KEYS = ("pixel_values", "image_grid_thw")


def doc_split(ids, which: str):
    """Seeded dev split for knob selection: dev = DEV_N ids, test = the rest, all = everything."""
    ids = sorted(ids)
    if which == "all":
        return set(ids)
    dev = set(random.Random(DEV_SEED).sample(ids, min(DEV_N, len(ids))))
    return dev if which == "dev" else set(ids) - dev


class DocTraceData(TraceData):
    """TraceData whose candidate units are the quotes (units with a quote sid, text or image); evidence.py's
    `is_video`/`vids` mean 'candidate evidence units'. `is_text` = the non-quote text units (question,
    instruction), used only by scope=global."""

    def __init__(self, arrays, rec):
        super().__init__(arrays, rec, None)
        quote = [u.get("sid") is not None and u["modality"] in ("text", "image") for u in self.units]
        self.is_video = np.array(quote)
        self.is_text = np.array([u["modality"] == "text" and not q for u, q in zip(self.units, quote)])
        self.vids = [u["uid"] for u, q in zip(self.units, quote) if q]


class QuoteAblation:
    """Hooks on the vision encoder (image quotes) and the input embeddings (text quotes).
    `masks[b]` = (image rows within one copy's vision output, text token positions) for batch copy b;
    `text_pool` = positions of all text-quote tokens (mean source for text replacement)."""

    def __init__(self, thinker):
        self.masks = None
        self.text_pool = None
        self.h_vis = thinker.visual.register_forward_hook(self._vis_hook)
        self.h_emb = thinker.get_input_embeddings().register_forward_hook(self._emb_hook)

    def _vis_hook(self, module, inputs, out):
        if self.masks is None:
            return None
        import torch
        t = out if torch.is_tensor(out) else getattr(out, "pooler_output", None)
        if t is None or not torch.is_tensor(t):
            raise RuntimeError("cannot find image embeddings in the vision encoder output")
        n = t.shape[0] // len(self.masks)
        new = t.clone()
        for b, (rows, _) in enumerate(self.masks):
            if rows is not None and len(rows):
                new[b * n + rows] = t[b * n:(b + 1) * n].mean(0, keepdim=True)
        if torch.is_tensor(out):
            return new
        out.pooler_output = new
        return out

    def _emb_hook(self, module, inputs, out):
        if self.masks is None or out.dim() != 3 or out.shape[0] != len(self.masks):
            return None
        new = out.clone()
        for b, (_, pos) in enumerate(self.masks):
            if pos is not None and len(pos):
                new[b, pos] = out[b, self.text_pool].mean(0, keepdim=True)
        return new


def batch_inputs(inputs, ids, B):
    """Repeat the (prompt + response) inputs B times (images repeated along their patch axis)."""
    import torch
    seq = torch.cat([inputs["input_ids"], ids], 1)
    full = {"input_ids": seq.repeat(B, 1)}
    full["attention_mask"] = torch.ones_like(full["input_ids"])
    for k, v in inputs.items():
        if k in ("input_ids", "attention_mask"):
            continue
        full[k] = v.repeat(B, *([1] * (v.dim() - 1))) if k in IMAGE_BATCH_KEYS and torch.is_tensor(v) else v
    return full


def causal_question(backend, ablate, q, rec, z, args):
    """All hidden-quote variants of one question -> result row (same layout as causal_video.py)."""
    import torch
    from omnitrace.data import build_doc_prompt

    thinker = getattr(backend, "thinker", None)
    prompt = build_doc_prompt(q, rec.get("doc_prompt") or "plain")
    inputs, tl, meta = backend.prepare(q, prompt)
    n_prompt = int(inputs["input_ids"].shape[1])
    if n_prompt != rec["meta"]["n_prompt_tokens"] or len(tl.units) != len(rec["units"]):
        raise RuntimeError(f"input mismatch: {n_prompt} vs {rec['meta']['n_prompt_tokens']}")
    gen_ids = [int(x) for x in z["gen_ids"]]
    T = len(gen_ids)
    dev = inputs["input_ids"].device

    td = DocTraceData(z, rec)
    attach(td, args.ext_store, rec["id"])
    quotes = list(td.vids)
    unit = {u["uid"]: u for u in tl.units}
    img_rows, a = {}, 0                       # image unit -> rows of the (one-copy) vision output
    for u in tl.units:
        if u["modality"] == "image":
            img_rows[u["uid"]] = list(range(a, a + u["n_tokens"]))
            a += u["n_tokens"]
    txt_pos = {uid: np.where(tl.pos_unit == uid)[0].tolist() for uid in quotes if unit[uid]["modality"] == "text"}
    if ablate is not None:
        ablate.text_pool = torch.tensor(sorted(p for ps in txt_pos.values() for p in ps), dtype=torch.long, device=dev)

    merge2 = int(inputs["pixel_values"].shape[0]) // max(1, sum(u["n_tokens"] for u in tl.units if u["modality"] == "image")) \
        if torch.is_tensor(inputs.get("pixel_values")) else 0     # Qwen only (pixel rows per image token)

    def patch_rows_of(units):                 # pixel rows (before the 2x2 merge) of the units' images
        return [r * merge2 + i for u in units for r in img_rows.get(u, []) for i in range(merge2)]

    def mask_of(units):
        rows = [r for u in units for r in img_rows.get(u, [])]
        pos = [p for u in units for p in txt_pos.get(u, [])]
        return (torch.tensor(rows, dtype=torch.long, device=dev) if rows else None,
                torch.tensor(pos, dtype=torch.long, device=dev) if pos else None)

    sent_tok = defaultdict(list)
    for k, si in td.tok_sent.items():
        if k < T and gen_ids[k] not in backend.stop_ids:
            sent_tok[si].append(k)
    content_sents = [si for si, ks in sorted(sent_tok.items()) if sum(td.tok_pos[k] in CONTENT_POS for k in ks) >= 3]
    resp_tok = sorted(k for ks in sent_tok.values() for k in ks)

    variants = {"orig": ()}
    rng = random.Random(zlib.crc32(rec["id"].encode()))
    rand_sets = {}

    def rand_k(k):
        if k not in rand_sets:
            rand_sets[k] = [tuple(sorted(rng.sample(quotes, k))) for _ in range(args.rand_seeds)]
            for i, s in enumerate(rand_sets[k]):
                variants[f"rand|{k}|{i}"] = s
                variants[f"keeprand|{k}|{i}"] = tuple(u for u in quotes if u not in set(s))

    plan = {}
    for spec in args.settings:
        cfg = parse_setting(spec)
        ranked, _ = rank_frames(td, cfg)
        name = setting_name(cfg) if spec != MASS_MEAN else "mass_mean"
        for k in args.ks:
            k = min(k, len(quotes) - 1)
            top = ranked[:k]
            variants[f"{name}|{k}|del_top"] = tuple(sorted(top))
            variants[f"{name}|{k}|del_bottom"] = tuple(sorted(ranked[-k:]))
            variants[f"{name}|{k}|keep_top"] = tuple(u for u in quotes if u not in set(top))
            rand_k(k)
            plan[f"{name}|{k}"] = k
    gold = {e["sid"] for e in (q.get("evidence") or [])}
    gt_units = [u for u in quotes if unit[u]["sid"] in gold]
    if gt_units and len(gt_units) < len(quotes) and not args.labels_only:
        variants["gt|del_top"] = tuple(sorted(gt_units))
        rand_k(len(gt_units))
        plan["gt"] = len(gt_units)
    sent_plan = {}
    for spec in args.sent_settings:
        cfg = parse_setting(spec)
        name = setting_name(cfg) if spec != MASS_MEAN else "mass_mean"
        for j in content_sents:
            ranked, _ = rank_frames(td, {**cfg, "sents": {j}})
            top = ranked[:args.sent_k]
            variants[f"sent|{name}|{j}"] = tuple(sorted(top))
            rand_k(len(top))
            sent_plan[f"{name}|{j}"] = len(top)

    # ---- the labels the fixed setting outputs (score >= tau x mean of the other units): hide exactly those
    label_sizes = {}
    for lkey, tau in ([] if args.probe else label_keys(args.label_tau, args.label_taus, "image_text")):
        lab = predicted_units(td, tau)
        label_sizes[lkey] = {"tau": tau, "response": len(lab), "sent": {}}
        if lab and len(lab) < len(quotes):
            variants[f"{lkey}|del_top"] = tuple(sorted(lab))
            variants[f"{lkey}|keep_top"] = tuple(u for u in quotes if u not in set(lab))
            rand_k(len(lab))
            plan[lkey] = len(lab)
        for j in ([] if args.no_sent else content_sents):
            lab = predicted_units(td, tau, {j})
            label_sizes[lkey]["sent"][str(j)] = len(lab)
            if lab and len(lab) < len(quotes):
                variants[f"sent|{lkey}|{j}"] = tuple(sorted(lab))
                rand_k(len(lab))
                sent_plan[f"{lkey}|{j}"] = len(lab)

    from omnitrace.ablation import black_row, blank_patches, blank_text, space_token_id
    ext_sizes = set_rows(td, args.ext_sets, quotes, variants, plan, sent_plan, rand_k, content_sents,
                         args.settings[:1], rank_frames, parse_setting, setting_name)
    probe = probe_variants(args.probe, quotes, rec["id"], args.probe_n) if args.probe else {}
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
                    if full.get("pixel_values") is not None:
                        pv = full["pixel_values"]
                        blank_patches(pv, [patch_rows_of(u) for u in chunk], black_row(backend.processor.image_processor, pv.shape[1]))
                    blank_text(full["input_ids"], [[p for x in u for p in txt_pos.get(x, [])] for u in chunk],
                               space_token_id(backend.tokenizer))
                else:
                    ablate.masks = [mask_of(u) for u in chunk]
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
                outs.append((tok_lp[b].cpu().numpy(), am[b].cpu().numpy()))
            del res, L, logits, lp
    by_set = dict(zip(uniq, outs))

    def summ(lp):
        return {"resp_lp": float(lp[resp_tok].mean()) if resp_tok else float("nan"),
                "sent_lp": {str(si): float(lp[ks].mean()) for si, ks in sent_tok.items()}}

    n_tok = {u: unit[u]["n_tokens"] for u in quotes}
    return {
        "id": rec["id"], "n_units": len(quotes), "correct": rec.get("correct"),
        "n_text_quotes": sum(unit[u]["modality"] == "text" for u in quotes),
        "quote_tokens": int(sum(n_tok.values())), "gold": sorted(gold),
        "argmax_agree": float((by_set[()][1] == np.array(gen_ids)).mean()),
        "fill": args.fill, "label_tau": {k: v["tau"] for k, v in label_sizes.items()}, "label_sizes": label_sizes,
        "content_sents": [str(j) for j in content_sents], "plan": plan, "sent_plan": sent_plan,
        "n_forward": len(uniq),
        "ext_sizes": ext_sizes, "probe": args.probe, "probe_sets": {k: list(v) for k, v in probe.items()},
        "sent_ntok": {str(si): len(ks) for si, ks in sent_tok.items()},
        "hidden_tokens": {name: int(sum(n_tok[u] for u in s)) for name, s in variants.items()},
        "hidden_images": {name: int(sum(unit[u]["modality"] == "image" for u in s)) for name, s in variants.items()},
        "results": {name: summ(by_set[s][0]) for name, s in variants.items()},
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
    qs = {q["id"]: q for q in json.loads(Path(args.questions).read_text())}
    keep = doc_split(qs, args.split)
    recs = [json.loads(l) for f in sorted(runp.glob("records*.jsonl")) for l in f.open() if l.strip()]
    recs = sorted({r["id"]: r for r in recs}.values(), key=lambda r: r["id"])
    recs = [r for r in recs if r["id"] in keep and (runp / "traces" / r["meta"]["trace_file"]).exists()]
    if args.n:
        random.Random(args.seed).shuffle(recs)
        recs = sorted(recs[: args.n], key=lambda r: r["id"])
    recs = [r for r in recs[args.shard_id::args.num_shards] if r["id"] not in done]
    print(f"{len(recs)} questions to run (split {args.split}, shard {args.shard_id}/{args.num_shards})",
          file=sys.stderr, flush=True)

    backend = get_backend(run_cfg["model"], args.model_path or run_cfg.get("model_path"),
                          image_max_pixels=run_cfg.get("image_max_pixels", 1024 * 28 * 28))
    args.ext_store = load_external(runp, ext_names(args.settings + args.sent_settings) + list(args.ext_sets))
    ablate = None if hasattr(backend, "hidden_logits") else QuoteAblation(backend.thinker)
    t_start = time.time()
    with out_path.open("a") as fout:
        bar = tqdm(recs, desc=f"causal-doc {args.tag} shard{args.shard_id}", file=sys.stderr, mininterval=30, ascii=True)
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


def summarize_doc(args):
    """causal_video.summarize on one split, plus the mean tokens / images each variant hides."""
    import tempfile
    runp = Path(args.run)
    rows = [json.loads(l) for f in tag_files(runp / "causal", args.tag) for l in f.open() if l.strip()]
    rows = list({r["id"]: r for r in rows}.values())
    if args.split != "all":
        keep = doc_split([q["id"] for q in json.loads(Path(args.questions).read_text())], args.split)
        rows = [r for r in rows if r["id"] in keep]
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "causal").mkdir()
        (Path(tmp) / "causal" / f"causal_{args.tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"split: {args.split}")
        summarize(argparse.Namespace(run=tmp, tag=args.tag, subset=args.subset, no_labels=args.no_labels))
        import shutil
        shutil.copy(Path(tmp) / "causal" / f"summary_{args.tag}_{args.subset}.csv",
                    runp / "causal" / f"summary_{args.tag}_{args.split}_{args.subset}.csv")
    acc = defaultdict(lambda: [[], []])
    for r in rows:
        for name, t in r["hidden_tokens"].items():
            key = name if not name.startswith(("rand|", "keeprand|", "sent|")) else name.rsplit("|", 1)[0]
            acc[key][0].append(t)
            acc[key][1].append(r["hidden_images"][name])
    print("\n=== hidden amount per variant (mean tokens, mean image quotes) ===")
    for key in sorted(acc):
        if key != "orig":
            print(f"{key:<32s} {np.mean(acc[key][0]):9.1f} {np.mean(acc[key][1]):6.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True)
    r.add_argument("--questions", required=True)
    r.add_argument("--model_path", default=None)
    r.add_argument("--split", choices=["dev", "test", "all"], default="all",
                   help="all = every question of --questions (default; use the dev/main files). dev/test = seeded "
                        "500 / rest of the file (for the full 2,000)")
    r.add_argument("--n", type=int, default=None, help="seeded random subset within the split")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--settings", nargs="+", default=["mass", MASS_MEAN, "v1", "soft_last"],
                   help="presets or 'k=v,...' specs; the default compares unit=sum (mass) and unit=mean")
    r.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3], help="numbers of hidden quotes")
    r.add_argument("--sent_settings", nargs="+", default=["mass", MASS_MEAN])
    r.add_argument("--sent_k", type=int, default=1, help="quotes hidden per sentence")
    r.add_argument("--rand_seeds", type=int, default=3)
    r.add_argument("--fill", choices=["blank", "mean"], default="blank", help="blank (default): empty placeholder of the same length at the input (black frames / silence / black image / space tokens); mean: mean-embedding replacement (masking-choice ablation)")
    r.add_argument("--label_tau", type=float, default=None,
                   help="score -> label rule constant (default: the modality's value in configs/label_tau.json)")
    r.add_argument("--label_taus", nargs="+", type=float, default=None,
                   help="sweep: label rows for each of these values (keys label@<tau>) instead of the saved one")
    r.add_argument("--labels_only", action="store_true", help="only the label rows (no budget / human-label rows)")
    r.add_argument("--no_sent", action="store_true", help="no sentence-level label rows (cheap tau sweep)")
    r.add_argument("--ext_sets", nargs="*", default=[], help="set-type baselines (<run>/baselines/<name>.jsonl), e.g. selfcite")
    r.add_argument("--probe", choices=["loo", "contextcite"], default=None,
                   help="only the ablation forwards a perturbation baseline is fitted on")
    r.add_argument("--probe_n", type=int, default=32, help="contextcite: random ablations per question")
    r.add_argument("--batch", type=int, default=4, help="hidden-quote variants per forward pass")
    r.add_argument("--tag", default="v1", help="output name; use a new tag for a new configuration")
    r.add_argument("--num_shards", type=int, default=1)
    r.add_argument("--shard_id", type=int, default=0)
    s = sub.add_parser("summarize")
    s.add_argument("--run", required=True)
    s.add_argument("--questions", required=True, help="for the dev/test split")
    s.add_argument("--tag", default="v1")
    s.add_argument("--split", choices=["dev", "test", "all"], default="all")
    s.add_argument("--subset", choices=["all", "correct", "wrong"], default="all")
    s.add_argument("--no_labels", action="store_true", help="leave out the label rows (e.g. rows made at an old tau)")
    args = ap.parse_args()
    if args.cmd == "run" and (args.labels_only or args.probe):
        args.settings, args.sent_settings = [], []
        args.labels_only = True
    run(args) if args.cmd == "run" else summarize_doc(args)


if __name__ == "__main__":
    main()
