"""Causal faithfulness test for temporal attribution (GPU), plus CPU summary.

Question: if we hide the frames an attribution setting points to, does the model's confidence in its
*own* text drop more than when we hide the same amount of random frames?

    # GPU, one process per shard (~3 per GPU), resumable across shard counts
    CUDA_VISIBLE_DEVICES=0 python -u scripts/causal_video.py run --run runs/nextgqa/qwen2.5-omni-7b \
        --questions $OMNITRACE_DATA/question_video.json --tag blank --num_shards 6 --shard_id 0
    # CPU summary of the returned jsonl
    python scripts/causal_video.py summarize --run runs/nextgqa/qwen2.5-omni-7b --tag blank [--subset correct]

Hiding frames (--fill blank, default): the pixels of the chosen 1-s frame groups are set to black before
the vision tower (an empty placeholder; omnitrace/ablation.py). --fill mean is the older variant: the vision
tower's output embeddings of those groups are replaced by the video's mean embedding (forward hook).
Token positions, prompt and TMRoPE time ids are unchanged either way.
The model's original response (saved gen_ids) is teacher-forced; we read the per-token log-probs.
Variants run in batches (--batch copies of the input per forward, each copy with its own hidden frames).

Response level, for every setting x budget (fraction of video duration):
  del_top     hide the setting's top frames                  comprehensiveness (big drop = faithful)
  del_bottom  hide its lowest-ranked frames (same count)
  keep_top    hide everything except its top frames           sufficiency (small drop = faithful)
  shared references: rand (hide random frames, same count), keeprand (keep random frames, same count),
  gt (hide the human-evidence frames; compared with random of the same count)
  label: hide the units the fixed setting labels (evidence.select_units, --label_tau) instead of a budget
Other methods (omnitrace/external.py): a setting "ext:<name>" ranks units by <run>/baselines/<name>.jsonl
  (same budget / sentence rows); --ext_sets <name> adds set rows (self-citation: hide the cited units) plus
  <setting>@<name> (the first --settings entry's top units of the same count). --probe loo|contextcite writes
  only the ablation forwards those two baselines are fitted on.
  targets: resp_lp = mean log-prob of the response tokens; ans_lp / ans_pred = final answer letter
Sentence level, for --sent_settings at --sent_budget: for every content sentence j, hide the top frames of
  sentence j's OWN evidence and read
    own_drop   = drop of sentence j's mean token log-prob   (vs random frames of the same count -> lift)
    cross_drop = mean drop of the other content sentences   (specificity = own_drop - cross_drop)
  "label" rows: the same with sentence j's own predicted label set.
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

import numpy as np

from omnitrace.evidence import tag_files  # noqa: E402
from omnitrace.evidence import (TraceData, label_keys, parse_setting, predicted_units, rank_frames, setting_name,
                                split_ids, take_budget)
from omnitrace.external import attach, ext_names, load_external, probe_variants, set_rows
from omnitrace.metrics import question_gt, unit_overlap_labels

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM", "ADV"}


# ----------------------------------------------------------------------------- GPU part
class FrameAblation:
    """Forward hook on the vision tower. `masks[b]` = row indices (within one video) to replace in batch
    copy b by that copy's mean embedding; the vision output holds len(masks) copies back to back."""

    def __init__(self, visual):
        self.masks = None
        self.handle = visual.register_forward_hook(self._hook)

    def _hook(self, module, inputs, out):
        if self.masks is None:
            return None
        import torch
        t = out if torch.is_tensor(out) else getattr(out, "pooler_output", None)
        if t is None or not torch.is_tensor(t):
            raise RuntimeError("cannot find vision embeddings in the vision tower output")
        n = t.shape[0] // len(self.masks)
        new = t.clone()
        for b, rows in enumerate(self.masks):
            if rows is not None and len(rows):
                new[b * n + rows] = t[b * n:(b + 1) * n].mean(0, keepdim=True)
        if torch.is_tensor(out):
            return new
        out.pooler_output = new
        return out


def char_offsets(tokenizer, gen_ids):
    offs, prev = [], ""
    for k in range(len(gen_ids)):
        cur = tokenizer.decode(gen_ids[: k + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        offs.append((len(prev), len(cur)))
        prev = cur
    return prev, offs


def answer_token_index(text, offs, letters):
    """Index of the generated token carrying the final answer letter (after the last 'answer')."""
    i = text.lower().rfind("answer")
    if i < 0:
        return None
    for j in range(i + 6, min(len(text), i + 40)):
        if text[j] in letters and (j + 1 == len(text) or not text[j + 1].isalpha()):
            for k, (a, b) in enumerate(offs):
                if a <= j < b:
                    return k
            return None
    return None


def batch_inputs(inputs, ids, B):
    """Repeat the (prompt + response) inputs B times along the batch axis."""
    import torch
    seq = torch.cat([inputs["input_ids"], ids], 1)
    full = {"input_ids": seq.repeat(B, 1)}
    full["attention_mask"] = torch.ones_like(full["input_ids"])
    for k, v in inputs.items():
        if k in ("input_ids", "attention_mask"):
            continue
        if k == "pixel_values_videos":
            full[k] = v.repeat(B, *([1] * (v.dim() - 1)))
        elif k in ("video_grid_thw", "video_second_per_grid") and torch.is_tensor(v):
            full[k] = v.repeat(B, *([1] * (v.dim() - 1)))
        else:
            full[k] = v
    return full


def run(args):
    import torch
    from tqdm import tqdm
    from omnitrace.backends import get_backend
    from omnitrace.data import LETTERS, build_mcqa_prompt

    for k, v in (("ext_sets", []), ("probe", None), ("probe_n", 32)):   # callers without the baseline args
        setattr(args, k, getattr(args, k, v))
    runp = Path(args.run)
    out_dir = runp / "causal"
    out_dir.mkdir(exist_ok=True)
    suffix = f".shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = out_dir / f"causal_{args.tag}{suffix}.jsonl"
    done = {json.loads(l)["id"] for f in tag_files(out_dir, args.tag) for l in f.open() if l.strip()}

    cfg_file = runp / "configs.jsonl"
    run_cfg = json.loads(cfg_file.read_text().splitlines()[0]) if cfg_file.exists() else \
        json.loads((runp / "config.json").read_text())
    qs = {q["id"]: q for q in json.loads(Path(args.questions).read_text())}
    recs = [json.loads(l) for f in sorted(runp.glob("records*.jsonl")) for l in f.open() if l.strip()]
    recs = sorted({r["id"]: r for r in recs}.values(), key=lambda r: r["id"])
    recs = [r for r in recs if r["id"] in qs and (runp / "traces" / r["meta"]["trace_file"]).exists()
            and qs[r["id"]].get("evidence")]
    random.Random(args.seed).shuffle(recs)
    recs = sorted(recs[: args.n], key=lambda r: r["id"])
    keep = split_ids([r["id"] for r in recs], "video", args.split, args.dev_frac)
    recs = [r for r in recs if r["id"] in keep][args.shard_id::args.num_shards]
    recs = [r for r in recs if r["id"] not in done]
    print(f"{len(recs)} questions to run (split {args.split}, shard {args.shard_id}/{args.num_shards})", file=sys.stderr, flush=True)

    backend = get_backend(run_cfg["model"], args.model_path or run_cfg.get("model_path"),
                          video_fps=run_cfg.get("video_fps", 2.0),
                          video_total_pixels=run_cfg.get("video_total_pixels"),
                          video_max_pixels=run_cfg.get("video_max_pixels"))
    thinker, tok = getattr(backend, "thinker", None), backend.tokenizer
    ext_store = load_external(runp, ext_names(args.settings + args.sent_settings) + list(args.ext_sets))
    from omnitrace.ablation import black_row, blank_patches
    own_hiding = hasattr(backend, "hidden_logits")      # MiniCPM-o hides units itself (embedding level)
    ablate = None if own_hiding else FrameAblation(thinker.visual)
    settings = [parse_setting(s) for s in args.settings]
    sent_settings = [parse_setting(s) for s in args.sent_settings]

    @torch.no_grad()
    def token_logprobs(inputs, gen_ids, n_prompt, unit_sets, rows_of, patch_rows_of, letter_ids, ans_idx):
        """Per hidden-frame set: (per-token log-probs [T], argmax ids [T], preferred answer letter)."""
        T = len(gen_ids)
        ids = torch.tensor([gen_ids], device=inputs["input_ids"].device)
        tgt = None
        out = []
        for i in range(0, len(unit_sets), args.batch):
            chunk = unit_sets[i:i + args.batch]
            if own_hiding:
                res = L = None
                logits = backend.hidden_logits(inputs, gen_ids, chunk, args.fill)
            else:
                full = batch_inputs(inputs, ids, len(chunk))
                if args.fill == "blank":
                    pv = full["pixel_values_videos"]
                    blank_patches(pv, [patch_rows_of(u) for u in chunk], black_row(backend.processor.image_processor, pv.shape[1]))
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
            if tgt is None:
                tgt = torch.tensor(gen_ids, device=lp.device)
            tok_lp = lp.gather(-1, tgt.view(1, T, 1).expand(lp.shape[0], T, 1))[..., 0]
            am = lp.argmax(-1)
            for b in range(lp.shape[0]):
                ap = int(lp[b, ans_idx, letter_ids].argmax()) if ans_idx is not None else None
                out.append((tok_lp[b].cpu().numpy(), am[b].cpu().numpy(), ap))
            del res, L, logits, lp
        return out

    t_start = time.time()
    with out_path.open("a") as fout:
        bar = tqdm(recs, desc=f"causal {args.tag} shard{args.shard_id}", file=sys.stderr, mininterval=30, ascii=True)
        for i_rec, rec in enumerate(bar, 1):
            try:
                q = qs[rec["id"]]
                prompt = build_mcqa_prompt(q, modality="video", style=rec.get("mcqa_prompt") or "plain")
                inputs, tl, meta = backend.prepare(q, prompt)
                n_prompt = int(inputs["input_ids"].shape[1])
                if n_prompt != rec["meta"]["n_prompt_tokens"] or len(tl.units) != len(rec["units"]):
                    raise RuntimeError(f"input mismatch: {n_prompt} vs {rec['meta']['n_prompt_tokens']}")
                z = np.load(runp / "traces" / rec["meta"]["trace_file"])
                gen_ids = [int(x) for x in z["gen_ids"]]
                letters = LETTERS[: len(q["options"])]
                letter_ids = [tok.convert_tokens_to_ids(l) for l in letters]
                text, offs = char_offsets(tok, gen_ids)
                ans_idx = answer_token_index(text, offs, letters)

                td = TraceData(z, rec, q["options"])
                attach(td, ext_store, rec["id"])
                dur = float(q["sources"][0]["duration"])
                per_group = meta["tokens_per_group"]
                gidx = {u["uid"]: int(u["label"][1:]) for u in tl.units if u["modality"] == "video"}
                all_v = list(gidx)
                dev = inputs["input_ids"].device

                def rows_of(units):
                    if not units:
                        return None
                    return torch.tensor([r for u in units for r in range(gidx[u] * per_group, (gidx[u] + 1) * per_group)],
                                        dtype=torch.long, device=dev)

                per_patch = int(inputs["pixel_values_videos"].shape[0]) // int(meta["video_grid_thw"][0]) \
                    if "pixel_values_videos" in inputs else 0     # Qwen only (pixel rows per frame group)

                def patch_rows_of(units):     # pixel rows (before the 2x2 merge) of the units' frame groups
                    return [r for u in units for r in range(gidx[u] * per_patch, (gidx[u] + 1) * per_patch)]

                # content tokens per sentence; content sentences = >=3 content words, not the "Answer: X" line
                sent_tok = defaultdict(list)
                for k, si in td.tok_sent.items():
                    if k < len(gen_ids) and gen_ids[k] not in backend.stop_ids:
                        sent_tok[si].append(k)
                content_sents = [si for si, ks in sorted(sent_tok.items())
                                 if sum(td.tok_pos[k] in CONTENT_POS for k in ks) >= 3
                                 and not rec["sentences"][si]["text"].lower().startswith("answer")]
                resp_tok = sorted(k for ks in sent_tok.values() for k in ks)

                # ---- all hidden-frame sets of this question (deduplicated below)
                variants = {"orig": ()}
                rng = random.Random(zlib.crc32(rec["id"].encode()))
                rand_sets = {}

                def rand_k(k):
                    if k not in rand_sets:
                        rand_sets[k] = [tuple(sorted(rng.sample(all_v, k))) for _ in range(args.rand_seeds)]
                        for i, s in enumerate(rand_sets[k]):
                            variants[f"rand|{k}|{i}"] = s
                            variants[f"keeprand|{k}|{i}"] = tuple(u for u in all_v if u not in set(s))

                plan = {}
                for cfg in settings:
                    ranked, _ = rank_frames(td, cfg)
                    name = setting_name(cfg)
                    for b in args.budgets:
                        top = take_budget(ranked, td.units, dur, b)
                        k = len(top)
                        variants[f"{name}|{b}|del_top"] = tuple(sorted(top))
                        variants[f"{name}|{b}|del_bottom"] = tuple(sorted(ranked[-k:]))
                        variants[f"{name}|{b}|keep_top"] = tuple(u for u in all_v if u not in set(top))
                        rand_k(k)
                        plan[f"{name}|{b}"] = k
                gsid, gt, _, _ = question_gt(q)
                vids, labels = unit_overlap_labels(td.units, gsid, gt)
                gt_units = [u for u, l in zip(vids, labels) if l]
                if gt_units and not args.labels_only:
                    variants["gt|del_top"] = tuple(sorted(gt_units))
                    rand_k(len(gt_units))
                    plan["gt"] = len(gt_units)
                sent_plan = {}
                for cfg in sent_settings:
                    name = setting_name(cfg)
                    for j in content_sents:
                        ranked, _ = rank_frames(td, {**cfg, "sents": {j}})
                        top = take_budget(ranked, td.units, dur, args.sent_budget)
                        variants[f"sent|{name}|{j}"] = tuple(sorted(top))
                        rand_k(len(top))
                        sent_plan[f"{name}|{j}"] = len(top)

                # ---- the labels the fixed setting outputs (score >= tau x mean of the other units): hide exactly those
                label_sizes = {}
                for lkey, tau in ([] if args.probe else label_keys(args.label_tau, args.label_taus, "video")):
                    lab = predicted_units(td, tau)
                    label_sizes[lkey] = {"tau": tau, "response": len(lab), "sent": {}}
                    if lab and len(lab) < len(all_v):
                        variants[f"{lkey}|del_top"] = tuple(sorted(lab))
                        variants[f"{lkey}|keep_top"] = tuple(u for u in all_v if u not in set(lab))
                        rand_k(len(lab))
                        plan[lkey] = len(lab)
                    for j in ([] if args.no_sent else content_sents):
                        lab = predicted_units(td, tau, {j})
                        label_sizes[lkey]["sent"][str(j)] = len(lab)
                        if lab and len(lab) < len(all_v):
                            variants[f"sent|{lkey}|{j}"] = tuple(sorted(lab))
                            rand_k(len(lab))
                            sent_plan[f"{lkey}|{j}"] = len(lab)

                ext_sizes = set_rows(td, args.ext_sets, all_v, variants, plan, sent_plan, rand_k, content_sents,
                                     args.settings[:1], rank_frames, parse_setting, setting_name)
                probe = probe_variants(args.probe, all_v, rec["id"], args.probe_n) if args.probe else {}
                variants.update(probe)

                uniq = sorted(set(variants.values()))
                outs = token_logprobs(inputs, gen_ids, n_prompt, [list(u) for u in uniq], rows_of, patch_rows_of,
                                      letter_ids, ans_idx)
                by_set = dict(zip(uniq, outs))

                def summ(lp, ap):
                    d = {"resp_lp": float(lp[resp_tok].mean()) if resp_tok else float("nan"),
                         "sent_lp": {str(si): float(lp[ks].mean()) for si, ks in sent_tok.items()}}
                    if ans_idx is not None:
                        d["ans_lp"] = float(lp[ans_idx])
                        d["ans_pred"] = ap
                    return d

                res = {name: summ(by_set[s][0], by_set[s][2]) for name, s in variants.items()}
                agree = float((by_set[()][1] == np.array(gen_ids)).mean())
                fout.write(json.dumps({
                    "id": rec["id"], "n_frames": len(all_v), "duration": dur, "correct": rec["correct"],
                    "qtype": q.get("meta", {}).get("qtype_group"), "argmax_agree": agree,
                    "fill": args.fill, "label_tau": {k: v["tau"] for k, v in label_sizes.items()}, "label_sizes": label_sizes,
                    "content_sents": [str(j) for j in content_sents], "plan": plan, "sent_plan": sent_plan,
                    "ext_sizes": ext_sizes, "probe": args.probe, "probe_sets": {k: list(v) for k, v in probe.items()},
                    "sent_ntok": {str(si): len(ks) for si, ks in sent_tok.items()},
                    "n_forward": len(uniq), "results": res}) + "\n")
                fout.flush()
                el = time.time() - t_start
                tqdm.write(f"[{i_rec}/{len(recs)}] {rec['id']}: {len(uniq)} variants, argmax_agree {agree:.3f} | "
                           f"elapsed {el / 60:.1f} min, ETA {el / i_rec * (len(recs) - i_rec) / 60:.1f} min",
                           file=sys.stderr)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                tqdm.write(f"[OOM] {rec['id']} (try a smaller --batch)", file=sys.stderr)
            except Exception:
                tqdm.write(f"[error] {rec['id']}\n{traceback.format_exc()[-1500:]}", file=sys.stderr)


# ----------------------------------------------------------------------------- CPU summary
def _boot(x, rng, n=1000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarize(args):
    runp = Path(args.run)
    rows = [json.loads(l) for f in tag_files(runp / "causal", args.tag) for l in f.open() if l.strip()]
    rows = list({r["id"]: r for r in rows}.values())                 # dedup across shard files
    split = getattr(args, "split", "all")
    if split != "all":                                               # dev / test of the tau selection
        keep = split_ids([r["id"] for r in rows], getattr(args, "modality", "video"), split, getattr(args, "dev_frac", None))
        rows = [r for r in rows if r["id"] in keep]
    if args.subset != "all":
        rows = [r for r in rows if bool(r["correct"]) == (args.subset == "correct")]
    if not rows:
        sys.exit("no rows")
    if getattr(args, "no_labels", False):                            # e.g. a first pass made at an old tau
        for r in rows:
            r["plan"] = {k: v for k, v in r["plan"].items() if not k.startswith("label")}
            r["sent_plan"] = {k: v for k, v in r.get("sent_plan", {}).items() if not k.startswith("label")}
    print(f"{len(rows)} questions (subset: {args.subset}, split: {split}); teacher-forced argmax == generated token on "
          f"{100 * np.mean([r['argmax_agree'] for r in rows]):.2f}% of positions")
    rng = np.random.default_rng(0)
    out_rows = []

    # ---- response level
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        res, o = r["results"], r["results"]["orig"]
        for key, k in r["plan"].items():
            rand = [res[f"rand|{k}|{i}"] for i in range(10) if f"rand|{k}|{i}" in res]
            keeprand = [res[f"keeprand|{k}|{i}"] for i in range(10) if f"keeprand|{k}|{i}" in res]
            top = res[f"{key}|del_top"]
            for metric in ("resp_lp", "ans_lp"):
                if metric not in o:
                    continue
                d = lambda x: o[metric] - x[metric]
                rd = float(np.mean([d(x) for x in rand]))
                a = acc[(key, metric)]
                a["del_top"].append(d(top)); a["del_rand"].append(rd); a["lift"].append(d(top) - rd)
                if f"{key}|del_bottom" in res:
                    a["del_bottom"].append(d(res[f"{key}|del_bottom"]))
                if f"{key}|keep_top" in res and keeprand:
                    kr = float(np.mean([d(x) for x in keeprand]))
                    kt = d(res[f"{key}|keep_top"])
                    a["keep_top"].append(kt); a["keep_rand"].append(kr); a["suff_gain"].append(kr - kt)
            if "ans_pred" in o and rand:
                f = acc[(key, "flip")]
                f["del_top"].append(float(top["ans_pred"] != o["ans_pred"]))
                f["del_rand"].append(float(np.mean([x["ans_pred"] != o["ans_pred"] for x in rand])))
    for (key, metric), a in sorted(acc.items()):
        row = {"level": "response", "setting|budget": key, "metric": metric, "n": len(a["del_top"])}
        for v in ("del_top", "del_rand", "lift", "del_bottom", "keep_top", "keep_rand", "suff_gain"):
            if a.get(v):
                row[v] = float(np.mean(a[v]))
        for v in ("lift", "suff_gain"):
            if a.get(v):
                row[f"{v}_ci_low"], row[f"{v}_ci_high"] = _boot(a[v], rng)
        if a.get("lift"):
            row["lift_pos%"] = 100 * float(np.mean(np.array(a["lift"]) > 0))
        out_rows.append(row)

    # ---- sentence level
    sacc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        res, o = r["results"], r["results"]["orig"]
        for key, k in r.get("sent_plan", {}).items():
            name, j = key.rsplit("|", 1)
            hid = res[f"sent|{key}"]
            rand = [res[f"rand|{k}|{i}"] for i in range(10) if f"rand|{k}|{i}" in res]
            own = o["sent_lp"][j] - hid["sent_lp"][j]
            rnd = float(np.mean([o["sent_lp"][j] - x["sent_lp"][j] for x in rand]))
            others = [s for s in r["content_sents"] if s != j]
            a = sacc[name]
            a["own_drop"].append(own); a["rand_drop"].append(rnd); a["lift"].append(own - rnd)
            if others:
                cross = float(np.mean([o["sent_lp"][s] - hid["sent_lp"][s] for s in others]))
                a["cross_drop"].append(cross); a["specificity"].append(own - cross)
    for name, a in sorted(sacc.items()):
        row = {"level": "sentence", "setting|budget": name, "metric": "sent_lp", "n": len(a["lift"])}
        for v in ("own_drop", "rand_drop", "lift", "cross_drop", "specificity"):
            if a.get(v):
                row[v] = float(np.mean(a[v]))
        for v in ("lift", "specificity"):
            if a.get(v):
                row[f"{v}_ci_low"], row[f"{v}_ci_high"] = _boot(a[v], rng)
        row["lift_pos%"] = 100 * float(np.mean(np.array(a["lift"]) > 0))
        out_rows.append(row)

    print("\nDrop = log-prob(original) - log-prob(after hiding frames); larger = the hidden frames mattered more.")
    print("lift = drop(top frames) - drop(random frames, same count)            > 0: better than chance")
    print("suff_gain = drop(keep only random frames) - drop(keep only top frames) > 0: top frames alone preserve more")
    print("specificity = own-sentence drop - other-sentences drop               > 0: a sentence's frames matter for THAT sentence")
    print("'flip' rows: answer-change rates. 'gt' = the human-evidence frames.")
    layouts = (("response", ["metric", "n", "del_top", "del_rand", "lift", "lift_ci_low", "lift_ci_high", "lift_pos%",
                             "del_bottom", "keep_top", "keep_rand", "suff_gain", "suff_gain_ci_low", "suff_gain_ci_high"]),
               ("sentence", ["metric", "n", "own_drop", "rand_drop", "lift", "lift_ci_low", "lift_ci_high", "lift_pos%",
                             "cross_drop", "specificity", "specificity_ci_low", "specificity_ci_high"]))
    for level, cols in layouts:
        print(f"\n=== {level} level ===")
        print(f"{'setting|budget':<28s} " + " ".join(f"{c[:11]:>11s}" for c in cols))
        for r in out_rows:
            if r["level"] == level:
                cells = [(f"{r[c]:11.4f}" if isinstance(r.get(c), float) else f"{str(r.get(c, '')):>11s}") for c in cols]
                print(f"{str(r['setting|budget'])[:28]:<28s} " + " ".join(cells))
    import csv
    path = runp / "causal" / f"summary_{args.tag}{'' if split == 'all' else '_' + split}_{args.subset}.csv"
    keys = ["level", "setting|budget", "metric", "n"] + sorted({k for r in out_rows for k in r} -
                                                               {"level", "setting|budget", "metric", "n"})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwritten {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True)
    r.add_argument("--questions", required=True)
    r.add_argument("--model_path", default=None)
    r.add_argument("--n", type=int, default=200, help="questions (seeded random subset of the run; nested)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--settings", nargs="+", default=["mass", "v1", "soft_last"])
    r.add_argument("--budgets", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    r.add_argument("--sent_settings", nargs="+", default=["mass", "v1"])
    r.add_argument("--sent_budget", type=float, default=0.2)
    r.add_argument("--rand_seeds", type=int, default=3)
    r.add_argument("--fill", choices=["blank", "mean"], default="blank", help="blank (default): empty placeholder of the same length at the input (black frames / silence / black image / space tokens); mean: mean-embedding replacement (masking-choice ablation)")
    r.add_argument("--label_tau", type=float, default=None,
                   help="score -> label rule constant (default: the modality's value in configs/label_tau.json)")
    r.add_argument("--label_taus", nargs="+", type=float, default=None,
                   help="sweep: label rows for each of these values (keys label@<tau>) instead of the saved one")
    r.add_argument("--labels_only", action="store_true", help="only the label rows (no budget / human-label rows)")
    r.add_argument("--no_sent", action="store_true", help="no sentence-level label rows (cheap tau sweep)")
    r.add_argument("--split", choices=["auto", "all", "dev", "test"], default="auto",
                   help="dev = seeded fraction for choosing tau, test = the rest; auto = test when this modality's "
                        "tau was chosen on such a dev split (configs/label_tau.json), else all")
    r.add_argument("--dev_frac", type=float, default=None)
    r.add_argument("--ext_sets", nargs="*", default=[], help="set-type baselines (<run>/baselines/<name>.jsonl), e.g. selfcite")
    r.add_argument("--probe", choices=["loo", "contextcite"], default=None,
                   help="only the ablation forwards a perturbation baseline is fitted on")
    r.add_argument("--probe_n", type=int, default=32, help="contextcite: random ablations per question")
    r.add_argument("--batch", type=int, default=4, help="hidden-frame variants per forward pass")
    r.add_argument("--tag", default="v2", help="output name; use a new tag for a new configuration")
    r.add_argument("--num_shards", type=int, default=1)
    r.add_argument("--shard_id", type=int, default=0)
    s = sub.add_parser("summarize")
    s.add_argument("--run", required=True)
    s.add_argument("--tag", default="v2")
    s.add_argument("--subset", choices=["all", "correct", "wrong"], default="all")
    s.add_argument("--no_labels", action="store_true", help="leave out the label rows (e.g. rows made at an old tau)")
    s.add_argument("--split", choices=["all", "dev", "test"], default="all", help="dev / test of the tau selection")
    s.add_argument("--dev_frac", type=float, default=None)
    args = ap.parse_args()
    if args.cmd == "run" and (args.labels_only or args.probe):
        args.settings, args.sent_settings = [], []
        args.labels_only = True
    run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    main()
