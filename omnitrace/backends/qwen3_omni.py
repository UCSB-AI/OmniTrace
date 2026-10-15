"""Qwen3-Omni-30B-A3B backend (thinker only; MoE LLM with SDPA + last-row attention capture).

Same input handling as the Qwen2.5-Omni backend (neutral system prompt, qwen_omni_utils, use_audio_in_video=False,
greedy decoding), with what Qwen3-Omni changes:
  vision: 16-px patches, 2x2 merge -> one LLM token per 32x32 px (Qwen2.5: 28x28). qwen_omni_utils is called with
    image_patch_size=16 so its resizing already lands inside the processor's own bounds (otherwise the processor
    resizes the frames a second time; e.g. small frames are upscaled to 128 tokens).
  video: frames sampled at `fps` (default 2) -> 2 frames per temporal group, spg = 2 / fps seconds per group.
    The true sampling fps is passed to the processor: without it the processor assumes fps 1 and the time ids
    (13 per second) would stretch the video 2x. Group g covers [g * spg, (g + 1) * spg] s = one time unit.
  audio: Whisper log-mel (128 bins, 100 frames/s) -> the encoder works in 1-s chunks (100 frames) and emits 13 tokens
    per full chunk (fewer for the last, partial one). One time unit = round(audio_unit_sec) chunks (default 1 s).
  DeepStack: the vision encoder also feeds features of 3 intermediate layers into the first LLM layers at the
    visual positions; hiding a frame group / image replaces those too.
  LLM: 48 layers, 32 query heads (4 KV heads), 128 experts (8 active).
Text units (MCQA prompt and MMDocRAG quotes) are aligned exactly as for Qwen2.5-Omni.

Hiding units for the causal tests (`hidden_logits`; the LLM runs on input embeddings, as for MiniCPM-o):
  blank (default): frame group / image -> the vision encoder's output (and DeepStack features) for black pixels of
           the same grid. The encoder attends within one frame group / one image only and its position embeddings
           do not depend on time, so this equals blackening the pixels.
           audio -> digital silence (the clip's log-mel floor, max - 2 after Whisper's normalisation) in the hidden
           chunks; the audio encoder is rerun per variant (its attention spans 8-s windows).
           text -> space tokens.
  mean:    rows (and DeepStack rows) -> the mean over the question's frames / images / audio tokens; text -> the
           mean input embedding of the text-quote tokens. zero: audio rows -> 0.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..capture import AttentionRecorder
from ..data import DocPrompt, Prompt, media_path
from ..timeline import AUDIO, IMAGE, TEXT, VIDEO, Timeline

DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."      # as for Qwen2.5-Omni


def audio_chunk_tokens(mel_len: int, n_window: int = 50) -> List[int]:
    """Audio tokens per encoder chunk (n_window * 2 mel frames = 1 s): 13 per full chunk, fewer for the last."""
    chunk = n_window * 2
    out = [13] * (mel_len // chunk)
    r = mel_len % chunk
    if r:
        out.append((((r - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1)
    return out


class Qwen3OmniBackend:
    name = "qwen3-omni"

    def __init__(self, model_path: str = DEFAULT_MODEL, device_map: str = "auto",
                 video_fps: float = 2.0, video_total_pixels: Optional[int] = None,
                 video_max_pixels: Optional[int] = None, audio_unit_sec: float = 1.0,
                 image_max_pixels: Optional[int] = 1024 * 28 * 28, dtype=torch.bfloat16):
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.model_path = model_path
        self.video_fps = video_fps
        self.video_total_pixels = video_total_pixels
        self.video_max_pixels = video_max_pixels
        self.image_max_pixels = image_max_pixels
        self.audio_unit_chunks = int(round(audio_unit_sec))
        if self.audio_unit_chunks < 1 or abs(audio_unit_sec - self.audio_unit_chunks) > 1e-6:
            raise ValueError(f"audio_unit_sec must be a whole number of seconds (encoder chunks), got {audio_unit_sec}")
        kw = dict(device_map=device_map, attn_implementation="sdpa", enable_audio_output=False)
        try:
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_path, dtype=dtype, **kw)
        except TypeError:   # older transformers: torch_dtype
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, **kw)
        if hasattr(model, "disable_talker"):
            model.disable_talker()
        model.eval()
        self.model = model
        self.thinker = model.thinker
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        conv = self.tokenizer.convert_tokens_to_ids
        self.video_token_id = conv("<|video_pad|>")
        self.image_token_id = conv("<|image_pad|>")
        self.audio_token_id = conv("<|audio_pad|>")
        self.vision_end_id = conv("<|vision_end|>")
        self.audio_end_id = conv("<|audio_end|>")
        self.im_end_id = conv("<|im_end|>")
        self.stop_ids = {self.im_end_id, conv("<|endoftext|>")}
        cfg = self.thinker.config
        self.patch_size = cfg.vision_config.patch_size
        self.merge = cfg.vision_config.spatial_merge_size
        self.n_window = cfg.audio_config.n_window
        # DeepStack mask layout: transformers 4.57 takes [B, N, 1] (it strips the last dim), 5.x takes [B, N]
        import inspect
        self._deep_mask_3d = "[..., 0]" in inspect.getsource(type(self.thinker.model)._deepstack_process)
        ids = self.tokenizer.encode(" ", add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"expected one token for a space, got {ids}")
        self.space_id = ids[0]

    @property
    def device(self):
        return self.thinker.get_input_embeddings().weight.device

    def decoder_layers(self):
        return self.thinker.model.layers

    # ------------------------------------------------------------------ inputs
    def build_messages(self, sample: Dict[str, Any], prompt: Prompt) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        system = {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        if isinstance(prompt, DocPrompt):
            for part in prompt.parts:
                if part["type"] == "text":
                    content.append({"type": "text", "text": part["text"]})
                else:
                    item = {"type": "image", "image": media_path(part)}
                    if self.image_max_pixels:
                        item["max_pixels"] = self.image_max_pixels
                    content.append(item)
            return [system, {"role": "user", "content": content}]
        for src in sample["sources"]:
            if src["modality"] == "audio":
                content.append({"type": "audio", "audio": media_path(src)})
                continue
            if src["modality"] != "video":
                raise NotImplementedError(f"modality {src['modality']} not wired yet")
            item = {"type": "video", "video": media_path(src), "fps": self.video_fps}
            if self.video_total_pixels:
                item["total_pixels"] = self.video_total_pixels
            if self.video_max_pixels:
                item["max_pixels"] = self.video_max_pixels
            content.append(item)
        content.append({"type": "text", "text": prompt.text})
        return [system, {"role": "user", "content": content}]

    def prepare(self, sample: Dict[str, Any], prompt: Prompt) -> Tuple[Dict[str, Any], Timeline, Dict[str, Any]]:
        from qwen_omni_utils import process_mm_info

        messages = self.build_messages(sample, prompt)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        try:
            audios, images, videos, video_kwargs = process_mm_info(
                messages, use_audio_in_video=False, return_video_kwargs=True, image_patch_size=self.patch_size)
        except TypeError as e:
            raise RuntimeError("qwen-omni-utils is too old for Qwen3-Omni (needs image_patch_size; >= 0.0.9)") from e
        fps_list = video_kwargs.get("fps") or []
        kwargs = dict(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
                      use_audio_in_video=False)
        if videos:
            if len(set(fps_list)) != 1:
                raise RuntimeError(f"expected one sampling fps, got {fps_list}")
            kwargs["fps"] = float(fps_list[0])      # sets video_second_per_grid = 2 / fps (the time ids)
        inputs = dict(self.processor(**kwargs).to(self.device))
        for k in ("pixel_values", "pixel_values_videos", "input_features"):   # not video_second_per_grid (time ids)
            if inputs.get(k) is not None:
                inputs[k] = inputs[k].to(self.thinker.dtype)
        if isinstance(prompt, DocPrompt):
            tl, meta, hide = self.build_doc_timeline(inputs, prompt, text)
        else:
            tl, meta, hide = self.build_timeline(inputs, sample, prompt, fps_list)
        inputs["_hide"] = hide
        inputs["_tl"] = tl
        return inputs, tl, meta

    def _mcqa_text_units(self, ids, tl, hide, prompt: Prompt, meta) -> None:
        """Text units: the user prompt tokens between the last media end tag and <|im_end|>."""
        n = len(ids)
        ends = [i for i, x in enumerate(ids) if x in (self.vision_end_id, self.audio_end_id)]
        if not ends:
            return
        start = max(ends) + 1
        end = next(i for i in range(start, n) if ids[i] == self.im_end_id)
        enc = self.tokenizer(prompt.text, add_special_tokens=False, return_offsets_mapping=True)
        if enc["input_ids"] == ids[start:end]:
            offs = enc["offset_mapping"]
            for u in prompt.units:
                pos = [start + k for k, (a, b) in enumerate(offs)
                       if b > a and a < u["end"] and b > u["start"] and prompt.text[a:b].strip()]
                if pos:
                    uid = tl.add_unit(np.array(pos), TEXT, u["label"])
                    hide[uid] = {"text": pos}
            meta["text_units_aligned"] = True
        else:
            uid = tl.add_unit(np.arange(start, end), TEXT, "prompt")
            hide[uid] = {"text": list(range(start, end))}
            meta["text_units_aligned"] = False

    def build_timeline(self, inputs, sample, prompt: Prompt, fps_list):
        ids = inputs["input_ids"][0].tolist()
        n = len(ids)
        tl = Timeline.empty(n)
        hide: Dict[int, Dict[str, Any]] = {}
        meta: Dict[str, Any] = {"n_prompt_tokens": n}

        # ---- video: one unit per temporal group
        vpos = np.array([i for i, x in enumerate(ids) if x == self.video_token_id], dtype=np.int64)
        if len(vpos):
            src = next(s for s in sample["sources"] if s["modality"] == "video")
            gt_, gh, gw = [int(x) for x in inputs["video_grid_thw"][0].tolist()]
            per_group = gh * gw // (self.merge * self.merge)
            assert per_group * gt_ == len(vpos), (per_group, gt_, len(vpos))
            spg = float(inputs["video_second_per_grid"][0])
            sample_fps = float(fps_list[0])
            duration = float(src.get("duration") or gt_ * spg)
            for g in range(gt_):
                t0 = g * spg
                t1 = min((g + 1) * spg, duration) if g == gt_ - 1 else (g + 1) * spg
                uid = tl.add_unit(vpos[g * per_group:(g + 1) * per_group], VIDEO, f"t{g}", sid=src["sid"],
                                  t0=round(t0, 3), t1=round(max(t1, t0), 3))
                hide[uid] = {"vgroups": [g]}
            meta.update(video_grid_thw=[gt_, gh, gw], tokens_per_group=per_group, second_per_grid=spg,
                        sample_fps=sample_fps, n_frames=gt_ * 2, video_duration=duration)

        # ---- audio: one unit per audio_unit_chunks encoder chunks (1 s each)
        apos = np.array([i for i, x in enumerate(ids) if x == self.audio_token_id], dtype=np.int64)
        if len(apos):
            src = next(s for s in sample["sources"] if s["modality"] == "audio")
            mel_len = int(inputs["feature_attention_mask"][0].sum())
            per_chunk = audio_chunk_tokens(mel_len, self.n_window)
            if sum(per_chunk) != len(apos):
                raise RuntimeError(f"audio tokens {len(apos)} != chunk layout {sum(per_chunk)}")
            starts = np.cumsum([0] + per_chunk)
            sec = self.n_window * 2 / 100.0
            duration = float(src.get("duration") or mel_len / 100.0)
            k = self.audio_unit_chunks
            for g, c0 in enumerate(range(0, len(per_chunk), k)):
                c1 = min(c0 + k, len(per_chunk))
                a, b = int(starts[c0]), int(starts[c1])
                t0 = c0 * sec
                t1 = min(c1 * sec, duration) if c1 == len(per_chunk) else c1 * sec
                uid = tl.add_unit(apos[a:b], AUDIO, f"t{g}", sid=src["sid"], t0=round(t0, 3), t1=round(max(t1, t0), 3))
                hide[uid] = {"audio_chunks": list(range(c0, c1)), "audio_tokens": list(range(a, b))}
            meta.update(n_audio_tokens=int(len(apos)), audio_unit_chunks=k, audio_unit_sec=k * sec,
                        audio_chunk_tokens=per_chunk, audio_mel_frames=mel_len, audio_duration=duration)
            inputs["_apos"] = apos

        self._mcqa_text_units(ids, tl, hide, prompt, meta)
        meta["timeline"] = tl.summary()
        return tl, meta, hide

    def build_doc_timeline(self, inputs, prompt: DocPrompt, template: str):
        """Image units (one per image) + text units, aligned through the chat-template string."""
        ids = inputs["input_ids"][0].tolist()
        n = len(ids)
        tl = Timeline.empty(n)
        hide: Dict[int, Dict[str, Any]] = {}
        grids = inputs["image_grid_thw"].tolist() if inputs.get("image_grid_thw") is not None else []
        counts = [int(t * h * w) // (self.merge * self.merge) for t, h, w in grids]

        enc = self.tokenizer(template, add_special_tokens=False, return_offsets_mapping=True)
        exp_ids, exp_offs, k = [], [], 0
        for tid, off in zip(enc["input_ids"], enc["offset_mapping"]):
            if tid == self.image_token_id:
                exp_ids += [tid] * counts[k]
                exp_offs += [(0, 0)] * counts[k]
                k += 1
            else:
                exp_ids.append(tid)
                exp_offs.append(tuple(off))
        if exp_ids != ids:
            raise RuntimeError("doc prompt alignment failed: template tokens != processor input_ids")

        img_parts = [p for p in prompt.parts if p["type"] == "image"]
        ipos = np.array([i for i, x in enumerate(ids) if x == self.image_token_id], dtype=np.int64)
        assert len(img_parts) == len(counts) and sum(counts) == len(ipos), (len(img_parts), counts, len(ipos))
        a = 0
        for j, (part, c) in enumerate(zip(img_parts, counts)):
            uid = tl.add_unit(ipos[a:a + c], IMAGE, part["sid"], sid=part["sid"])
            hide[uid] = {"images": [j]}
            a += c

        markup = "<|vision_start|><|image_pad|><|vision_end|>"
        user, base_of = "", []
        for part in prompt.parts:
            base_of.append(len(user))
            user += part["text"] if part["type"] == "text" else markup
        base = template.find(user)
        if base < 0:
            raise RuntimeError("doc prompt alignment failed: user content not found in the chat template")
        for part, off in zip(prompt.parts, base_of):
            for u in part.get("units", []):
                s0, s1 = base + off + u["start"], base + off + u["end"]
                pos = [i for i, (x0, x1) in enumerate(exp_offs)
                       if x1 > x0 and x0 < s1 and x1 > s0 and template[x0:x1].strip()]
                if pos:
                    uid = tl.add_unit(np.array(pos), TEXT, u["label"], sid=u["sid"])
                    hide[uid] = {"text": pos}
        meta = {"n_prompt_tokens": n, "image_grid_thw": grids, "image_tokens": counts,
                "text_units_aligned": True, "timeline": tl.summary()}
        return tl, meta, hide

    @staticmethod
    def model_inputs(inputs) -> Dict[str, Any]:
        return {k: v for k, v in inputs.items() if not k.startswith("_")}

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, inputs, max_new_tokens: int = 512, recorder=None, force_ids=None) -> Dict[str, Any]:
        """force_ids: emit these tokens instead of greedy ones (AT2 features on a saved response)."""
        n_prompt = int(inputs["input_ids"].shape[1])
        rec = recorder or AttentionRecorder(n_prompt)   # recorder: e.g. capture.HeadUnitRecorder (per-head, AT2)
        layers = self.decoder_layers()
        rec.attach(layers)
        rec.active = True
        t0 = time.time()
        force = {}
        if force_ids is not None:
            from transformers import LogitsProcessorList
            from omnitrace.capture import ForcedTokens
            force = {"logits_processor": LogitsProcessorList([ForcedTokens(force_ids)])}
            max_new_tokens = len(force_ids)
        try:
            out = self.thinker.generate(
                **self.model_inputs(inputs), use_audio_in_video=False, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                eos_token_id=sorted(self.stop_ids), pad_token_id=self.tokenizer.pad_token_id or self.im_end_id,
                return_dict_in_generate=True, **force)
        finally:
            rec.active = False
            rec.detach(layers)
        seq = out.sequences if hasattr(out, "sequences") else out
        gen_ids = seq[0, n_prompt:].tolist()
        A = rec.stacked()                                   # [T, L, n_prompt]
        T = min(len(gen_ids), A.shape[0])
        return {"gen_ids": gen_ids[:T], "A": A[:T], "gen_seconds": time.time() - t0,
                "n_steps_recorded": int(A.shape[0]), "n_generated": len(gen_ids)}

    # ------------------------------------------------------------------ encoders (both transformers 4.57 and 5.x)
    def _visual(self, pixels, grid_thw) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Vision encoder -> (merged tokens [n, D], DeepStack features [k] x [n, D])."""
        vis = self.thinker.visual
        out = vis(pixels.type(vis.dtype), grid_thw=grid_thw)
        if isinstance(out, tuple):
            return out[0], list(out[1])
        return out.pooler_output, list(out.deepstack_features)

    def _audio(self, feats, lens) -> torch.Tensor:
        """Audio encoder on valid log-mel frames feats [mels, sum(lens)] -> [tokens, D]."""
        out = self.thinker.audio_tower(feats, feature_lens=lens)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def _audio_mel(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = inputs["feature_attention_mask"]
        feats = inputs["input_features"].permute(0, 2, 1)[mask.bool()].permute(1, 0)   # [mels, frames]
        return feats, mask.sum(-1)

    @torch.no_grad()
    def _encode(self, inputs) -> Dict[str, Any]:
        """Prompt embeddings with the encoder outputs scattered in, and the DeepStack inputs (cached in `inputs`)."""
        if "_enc" in inputs:
            return inputs["_enc"]
        th = self.thinker
        ids = inputs["input_ids"][0]
        emb = th.get_input_embeddings()(ids)                              # [N, D]
        vis_mask = torch.zeros_like(ids, dtype=torch.bool)
        e: Dict[str, Any] = {}
        parts = []
        for kind, pix, grid, tok in (("image", "pixel_values", "image_grid_thw", self.image_token_id),
                                     ("video", "pixel_values_videos", "video_grid_thw", self.video_token_id)):
            if inputs.get(pix) is None:
                continue
            pool, deep = self._visual(inputs[pix], inputs[grid])
            m = ids == tok
            emb[m] = pool.to(emb.dtype)
            vis_mask |= m
            e[kind] = {"pool": pool.to(emb.dtype), "deep": [d.to(emb.dtype) for d in deep], "pos": m.nonzero()[:, 0]}
            parts.append(kind)
        if inputs.get("input_features") is not None:
            feats, lens = self._audio_mel(inputs)
            aud = self._audio(feats, lens).to(emb.dtype)
            m = ids == self.audio_token_id
            emb[m] = aud
            e["audio"] = {"emb": aud, "pos": m.nonzero()[:, 0], "feats": feats, "lens": lens}
        # DeepStack inputs over all visual positions in sequence order (images and videos interleaved)
        rank = torch.cumsum(vis_mask.long(), 0) - 1                       # position -> row among visual positions
        deep = None
        if parts:
            n_vis, D = int(vis_mask.sum()), emb.shape[-1]
            levels = len(e[parts[0]]["deep"])
            deep = [emb.new_zeros(n_vis, D) for _ in range(levels)]
            for kind in parts:
                rows = rank[e[kind]["pos"]]
                for lvl in range(levels):
                    deep[lvl][rows] = e[kind]["deep"][lvl]
        e.update(emb=emb, vis_mask=vis_mask, rank=rank, deep=deep)
        inputs["_enc"] = e
        return e

    @torch.no_grad()
    def prompt_embeds(self, inputs) -> torch.Tensor:
        """[1, N, D] input embeddings as the thinker builds them (encoder outputs scattered in)."""
        return self._encode(inputs)["emb"][None]

    def _black(self, grid: Tuple[int, int, int], dim: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encoder output for a black image / frame group of this grid (cached per grid)."""
        cache = self.__dict__.setdefault("_black_cache", {})
        if grid not in cache:
            from ..ablation import black_row
            t, h, w = grid
            row = black_row(self.processor.image_processor, dim).to(self.device)
            px = row.expand(t * h * w, dim).contiguous()
            pool, deep = self._visual(px, torch.tensor([list(grid)], device=self.device))
            dt = self.thinker.get_input_embeddings().weight.dtype
            cache[grid] = (pool.to(dt), [d.to(dt) for d in deep])
        return cache[grid]

    @torch.no_grad()
    def _hide_cache(self, inputs, fill: str) -> Dict[str, Any]:
        """Per-question replacement rows for each hideable piece (frame group, image, audio, text)."""
        key = f"_hide_{fill}"
        if key in inputs:
            return inputs[key]
        e = self._encode(inputs)
        c: Dict[str, Any] = {}
        if "video" in e:
            gt_, gh, gw = [int(x) for x in inputs["video_grid_thw"][0].tolist()]
            per = gh * gw // (self.merge * self.merge)
            c["vgroup_pos"] = [e["video"]["pos"][g * per:(g + 1) * per] for g in range(gt_)]
            if fill == "blank":
                pool, deep = self._black((1, gh, gw), inputs["pixel_values_videos"].shape[1])
                c["vfill"] = [(pool, deep)] * gt_
            else:
                v = e["video"]
                c["vfill_mean"] = (v["pool"].mean(0), [d.mean(0) for d in v["deep"]])
        if "image" in e:
            grids = [tuple(int(x) for x in g) for g in inputs["image_grid_thw"].tolist()]
            sizes = [t * h * w // (self.merge * self.merge) for t, h, w in grids]
            starts = np.cumsum([0] + sizes)
            c["img_pos"] = [e["image"]["pos"][starts[j]:starts[j + 1]] for j in range(len(grids))]
            if fill == "blank":
                c["ifill"] = [self._black(g, inputs["pixel_values"].shape[1]) for g in grids]
            else:
                v = e["image"]
                c["ifill_mean"] = (v["pool"].mean(0), [d.mean(0) for d in v["deep"]])
        if "audio" in e and fill == "mean":
            c["aud_mean"] = e["audio"]["emb"].mean(0)
        hide = inputs["_hide"]
        text_units = [u for u in inputs["_tl"].units if u["modality"] == "text"]
        pool = [p for u in text_units if u.get("sid") is not None for p in hide[u["uid"]]["text"]] or \
               [p for u in text_units for p in hide[u["uid"]]["text"]]
        if fill == "blank":
            c["text_fill"] = self.thinker.get_input_embeddings()(torch.tensor([self.space_id], device=self.device))[0]
        elif pool:
            c["text_fill"] = e["emb"][torch.tensor(pool, device=self.device)].mean(0)
        inputs[key] = c
        return c

    @torch.no_grad()
    def _blank_audio_embeds(self, inputs, chunk_sets: Sequence[List[int]]) -> List[torch.Tensor]:
        """Audio embeddings [n_audio, D] with the given 1-s chunks silenced in the log-mel input (one per set)."""
        a = self._encode(inputs)["audio"]
        feats, lens = a["feats"], a["lens"]
        n = int(lens.sum())
        floor = max(float(feats.max()) - 2.0, -1.5)       # Whisper: clamp at max - 8, then (x + 4) / 4
        per = self.n_window * 2
        x = feats.repeat(1, len(chunk_sets))                # B clips back to back along time
        for b, chunks in enumerate(chunk_sets):
            for c in chunks:
                x[:, b * n + c * per:b * n + min((c + 1) * per, n)] = floor
        out = self._audio(x, lens.repeat(len(chunk_sets)))
        return list(out.to(a["emb"].dtype).split(out.shape[0] // len(chunk_sets)))

    def _positions(self, inputs, T: int) -> torch.Tensor:
        """[4, 1, N + T] position ids of prompt + response (text row, then the 3 MRoPE rows), as in generate()."""
        key = f"_pos_{T}"
        if key not in inputs:
            ids = inputs["input_ids"]
            full = torch.cat([ids, ids.new_full((1, T), self.space_id)], 1)    # response tokens are text
            mask = torch.ones_like(full)
            fam = inputs.get("feature_attention_mask")
            pos, _ = self.thinker.get_rope_index(
                full, inputs.get("image_grid_thw"), inputs.get("video_grid_thw"), mask, False,
                fam.sum(-1) if fam is not None else None, inputs.get("video_second_per_grid"))
            text = torch.arange(full.shape[1], device=pos.device, dtype=pos.dtype).view(1, 1, -1)
            inputs[key] = torch.cat([text, pos], 0)      # float: video time ids are fractional (g * spg * 13)
        return inputs[key]

    @torch.no_grad()
    def hidden_logits(self, inputs, gen_ids: List[int], unit_sets: Sequence[Sequence[int]], fill: str = "blank"):
        """Teacher-forced logits for the response tokens, one batch copy per unit set (uids hidden in that copy).
        Returns [B, T, V]; position t predicts gen_ids[t]."""
        e = self._encode(inputs)
        c = self._hide_cache(inputs, fill)
        hide = inputs["_hide"]
        emb = e["emb"]
        N, T, B = emb.shape[0], len(gen_ids), len(unit_sets)
        dev = emb.device
        gen = self.thinker.get_input_embeddings()(torch.tensor(gen_ids, device=dev))
        x = torch.cat([emb, gen], 0).unsqueeze(0).repeat(B, 1, 1)
        deep = [d.unsqueeze(0).repeat(B, 1, 1) for d in e["deep"]] if e["deep"] is not None else None
        rank = e["rank"]

        def put(b, pos, rows, deep_rows):
            x[b, pos] = rows
            for lvl, d in enumerate(deep_rows):
                deep[lvl][b, rank[pos]] = d

        chunk_sets = {b: [ch for u in us for ch in hide[u].get("audio_chunks", [])] for b, us in enumerate(unit_sets)}
        chunk_sets = {b: s for b, s in chunk_sets.items() if s}
        if chunk_sets and fill == "blank":
            for b, a in zip(chunk_sets, self._blank_audio_embeds(inputs, list(chunk_sets.values()))):
                x[b, e["audio"]["pos"]] = a
        for b, us in enumerate(unit_sets):
            for u in us:
                h = hide[u]
                for g in h.get("vgroups", []):
                    pool, dp = c["vfill"][g] if fill == "blank" else c["vfill_mean"]
                    put(b, c["vgroup_pos"][g], pool, dp)
                for j in h.get("images", []):
                    pool, dp = c["ifill"][j] if fill == "blank" else c["ifill_mean"]
                    put(b, c["img_pos"][j], pool, dp)
                if "text" in h:
                    x[b, torch.tensor(h["text"], device=dev)] = c["text_fill"]
                if "audio_tokens" in h and fill != "blank":
                    rows = e["audio"]["pos"][torch.tensor(h["audio_tokens"], device=dev)]
                    x[b, rows] = c["aud_mean"] if fill == "mean" else 0
        pos = self._positions(inputs, T).expand(-1, B, -1)
        kw = {}
        if deep is not None:
            vm = torch.cat([e["vis_mask"], e["vis_mask"].new_zeros(T)])[None].expand(B, -1)
            vm = vm[..., None] if self._deep_mask_3d else vm
            kw = {"visual_pos_masks": vm, "deepstack_visual_embeds": [d.reshape(-1, d.shape[-1]) for d in deep]}
        out = self.thinker.model(inputs_embeds=x, position_ids=pos, attention_mask=torch.ones(B, N + T, dtype=torch.long, device=dev),
                                 use_cache=False, **kw)
        hs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return self.thinker.lm_head(hs[:, N - 1:N - 1 + T])
