"""MiniCPM-o 4.5 backend (trust_remote_code; Qwen3-8B LLM with SDPA + last-row attention capture).

Needs its own environment (requirements_minicpm.txt: transformers==4.51.0 as on the model card; the remote code
uses cache APIs that later versions removed). Input handling follows the official `MiniCPMO.chat`:
  one user turn, no system prompt, content items joined by "\n"; chat template with enable_thinking=False (adds
  an empty <think></think>) and, when audio is present, use_tts_template=True (chat() forces it for audio; the
  text answer then ends with <|tts_eos|>, added here as a stop token).
  Decoding: greedy with repetition_penalty 1.0, as for Qwen (chat()'s default is 3-beam search with penalty 1.2,
  which has no single attention trace).

Video: frames from minicpmo.utils.get_video_frame_audio_segments (the official sampler): one frame per second
  (frame g = the frame at g s). Its MAX_NUM_FRAMES is raised to `max_frames` (default 128) so every NExT-GQA video
  (<= 91 s) keeps 1 fps; the official cap of 64 would switch longer videos to 64 uniform frames. Each frame is one
  image with max_slice_nums=1 and use_image_id=False (the official video settings): 64 resampler tokens inside
  <image></image>. Frames carry no time ids; their order is the only time signal. One frame = one time unit [g, g+1) s.
Audio: librosa 16 kHz mono -> Whisper log-mel (100 frames/s) -> Whisper encoder (chunked attention, 1-s chunks)
  -> avg-pool 5 -> 10 tokens/s: token k covers [k * 0.1, (k + 1) * 0.1] s. Audio over 30 s is encoded in 30-s
  segments (300 tokens each). Units = runs of round(audio_unit_sec * 10) tokens (default 1 s = 10 tokens).
Images (MMDocRAG): official slicing (max_slice_nums 9, scale 448): a thumbnail plus up to 9 slices, 64 tokens each,
  with image ids (<image_id>k</image_id>, the processor default). One unit = the whole image (all its blocks).
Text units are aligned exactly: the processor's final string (template with the media placeholders expanded) is
  rebuilt and tokenized with offsets; its ids must equal the processor's input_ids.

Hiding units for the causal tests (`hidden_logits`). The LLM runs on input embeddings, so every fill is applied
there, computed as if the input itself had been edited:
  blank (default): frame / image -> the encoder output of a black image of the same size. The vision encoder and
           the resampler encode every image block on its own, so this equals blackening the pixels.
           audio -> silence (the clip's log-mel floor) in the hidden seconds; the audio encoder is rerun per
           variant (its chunked attention lets a second influence the later ones).
           text -> space tokens.
  mean:    rows -> the mean encoder output of the question's frames / images / audio; text -> the mean input
           embedding of the text-quote tokens (as --fill mean in the Qwen scripts). zero: audio rows -> 0.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..capture import AttentionRecorder
from ..data import DocPrompt, media_path
from ..timeline import AUDIO, IMAGE, TEXT, VIDEO, Timeline

DEFAULT_MODEL = "openbmb/MiniCPM-o-4_5"
AUDIO_TOKENS_PER_SEC = 10          # 100 mel frames/s -> conv stride 2 -> avg-pool 5
MEL_PER_TOKEN = 10
MEDIA_SPLIT = "(<image>./</image>|<audio>./</audio>)"     # the processor's own (unescaped) pattern
IMAGE_MARK, AUDIO_MARK = "<image>./</image>", "<audio>./</audio>"


class MiniCPMOBackend:
    name = "minicpm-o"

    def __init__(self, model_path: str = DEFAULT_MODEL, device: Optional[str] = None, max_frames: int = 128,
                 audio_unit_sec: float = 1.0, doc_max_slice_nums: int = 9, dtype=torch.bfloat16, **qwen_only):
        # qwen_only: video_fps / video_total_pixels / video_max_pixels / image_max_pixels are Qwen settings
        # (callers pass them for every model); MiniCPM-o uses the official sampler (1 fps) and slicing instead.
        from transformers import AutoModel

        if qwen_only.get("audio_in_video"):
            raise NotImplementedError("audio_in_video is implemented for Qwen2.5-Omni only")
        self.model_path = model_path
        self.max_frames = max_frames
        self.audio_unit_tokens = max(1, int(round(audio_unit_sec * AUDIO_TOKENS_PER_SEC)))
        self.doc_max_slice_nums = doc_max_slice_nums
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation="sdpa",
                                          torch_dtype=dtype, init_vision=True, init_audio=True, init_tts=False)
        model.eval().to(device)
        model.prepare_processor(processor=None, tokenizer=None)
        self.model = model
        self.llm = model.llm
        self.processor = model.processor
        self.tokenizer = self.processor.tokenizer
        conv = self.tokenizer.convert_tokens_to_ids
        self.im_end_id = conv("<|im_end|>")
        self.tts_eos_id = conv("<|tts_eos|>")
        self.stop_ids = {self.im_end_id, conv("<|endoftext|>"), self.tts_eos_id}
        self.space_id = self._space_id()

    def _space_id(self) -> int:
        ids = self.tokenizer.encode(" ", add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"expected one token for a space, got {ids}")
        return ids[0]

    @property
    def device(self):
        return self.llm.get_input_embeddings().weight.device

    def decoder_layers(self):
        return self.llm.model.layers

    # ------------------------------------------------------------------ inputs
    def build_content(self, sample: Dict[str, Any], prompt) -> Tuple[List[Any], str]:
        """Official content list (PIL images, 16 kHz arrays, strings) and the input kind (video|audio|doc)."""
        from PIL import Image

        if isinstance(prompt, DocPrompt):
            items = [p["text"] if p["type"] == "text" else Image.open(media_path(p)).convert("RGB")
                     for p in prompt.parts]
            return items, "doc"
        items, kind = [], None
        for src in sample["sources"]:
            if src["modality"] == "video":
                from minicpmo import utils as mu
                mu.MAX_NUM_FRAMES = self.max_frames          # read at call time by the sampler
                if float(src.get("duration") or 0) > self.max_frames:   # the sampler would switch to uniform frames
                    raise RuntimeError(f"video longer than max_frames={self.max_frames} s; raise max_frames")
                frames, _, _ = mu.get_video_frame_audio_segments(media_path(src), use_audio=False)
                items += frames
                kind = "video"
            elif src["modality"] == "audio":
                import librosa
                wav, _ = librosa.load(media_path(src), sr=16000, mono=True)
                items.append(wav)
                kind = "audio"
            else:
                raise NotImplementedError(f"modality {src['modality']} not wired yet")
        items.append(prompt.text)
        return items, kind

    def prepare(self, sample: Dict[str, Any], prompt) -> Tuple[Dict[str, Any], Timeline, Dict[str, Any]]:
        from PIL import Image

        items, kind = self.build_content(sample, prompt)
        images = [c for c in items if isinstance(c, Image.Image)]
        audios = [c for c in items if isinstance(c, np.ndarray)]
        content = "\n".join(IMAGE_MARK if isinstance(c, Image.Image) else AUDIO_MARK if isinstance(c, np.ndarray)
                            else c for c in items)
        template = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
            use_tts_template=bool(audios), enable_thinking=False)
        msn, use_id = (1, False) if kind == "video" else (self.doc_max_slice_nums, None)
        proc = self.processor([template], [images], [audios], [[0] * len(audios)], max_slice_nums=msn,
                              use_image_id=use_id, return_tensors="pt", max_length=None)
        image_sizes = proc.pop("image_sizes")[0] if "image_sizes" in proc else []
        proc = proc.to(self.device)
        inputs = dict(proc)
        ids = inputs["input_ids"][0].tolist()
        n = len(ids)

        # exact alignment: rebuild the processor's final string, tokenize it with offsets, compare the ids
        final, shift = self._expand(template, image_sizes, audios, msn, use_id)
        enc = self.tokenizer(final, return_offsets_mapping=True)
        if enc["input_ids"] != ids:
            raise RuntimeError("prompt alignment failed: rebuilt template tokens != processor input_ids")
        offs = [tuple(o) for o in enc["offset_mapping"]]
        base = template.find(content)
        if base < 0:
            raise RuntimeError("prompt alignment failed: user content not found in the chat template")

        def positions(t0: int, t1: int) -> np.ndarray:      # template char range -> token positions
            s0, s1 = shift(t0), shift(t1)
            return np.array([i for i, (a, b) in enumerate(offs) if b > a and a < s1 and b > s0 and final[a:b].strip()],
                            dtype=np.int64)

        bounds = [tuple(int(x) for x in b) for b in (inputs["image_bound"][0].tolist() if len(inputs["image_bound"][0]) else [])]
        abounds = [tuple(int(x) for x in b) for b in (inputs["audio_bounds"][0].tolist() if len(inputs["audio_bounds"][0]) else [])]
        tl = Timeline.empty(n)
        hide: Dict[int, Dict[str, Any]] = {}
        meta: Dict[str, Any] = {"n_prompt_tokens": n, "input_kind": kind, "text_units_aligned": True}

        if kind == "video":
            src = next(s for s in sample["sources"] if s["modality"] == "video")
            if len(bounds) != len(images):
                raise RuntimeError(f"{len(bounds)} image blocks for {len(images)} frames (expected one per frame)")
            dur = float(src.get("duration") or len(images))
            for g, (a, b) in enumerate(bounds):
                t1 = min(g + 1.0, dur) if g == len(bounds) - 1 else g + 1.0
                uid = tl.add_unit(np.arange(a, b), VIDEO, f"t{g}", sid=src["sid"], t0=float(g), t1=round(max(t1, g), 3))
                hide[uid] = {"blocks": [g]}
            meta.update(n_frames=len(images), tokens_per_group=bounds[0][1] - bounds[0][0] if bounds else 0,
                        sample_fps=1.0, second_per_grid=1.0, video_duration=dur, max_frames=self.max_frames)
        elif kind == "audio":
            src = next(s for s in sample["sources"] if s["modality"] == "audio")
            apos = np.concatenate([np.arange(a, b) for a, b in abounds]).astype(np.int64)
            step, k = 1.0 / AUDIO_TOKENS_PER_SEC, self.audio_unit_tokens
            dur = float(src.get("duration") or len(apos) * step)
            for g, a in enumerate(range(0, len(apos), k)):
                b = min(a + k, len(apos))
                t0, t1 = a * step, (min(b * step, dur) if b == len(apos) else b * step)
                uid = tl.add_unit(apos[a:b], AUDIO, f"t{g}", sid=src["sid"], t0=round(t0, 3), t1=round(max(t1, t0), 3))
                hide[uid] = {"audio_tokens": list(range(a, b))}
            meta.update(n_audio_tokens=int(len(apos)), audio_unit_tokens=k, audio_unit_sec=k * step,
                        audio_duration=dur, audio_segments=int(inputs["audio_features"].shape[0]))
            inputs["_apos"] = apos
        else:   # doc: one unit per image (thumbnail + slices), then the text units
            ip = self.processor.image_processor
            counts = []
            for size in image_sizes:
                grid = ip.get_sliced_grid(size, msn) if ip.slice_mode else None
                counts.append(1 + (grid[0] * grid[1] if grid else 0))
            if sum(counts) != len(bounds):
                raise RuntimeError(f"image blocks {counts} do not match {len(bounds)} image bounds")
            img_parts = [p for p in prompt.parts if p["type"] == "image"]
            j = 0
            for part, c in zip(img_parts, counts):
                pos = np.concatenate([np.arange(a, b) for a, b in bounds[j:j + c]])
                uid = tl.add_unit(pos, IMAGE, part["sid"], sid=part["sid"])
                hide[uid] = {"blocks": list(range(j, j + c))}
                j += c
            meta.update(image_blocks=counts, image_tokens=[int(sum(b - a for a, b in bounds[j0:j0 + c]))
                                                          for j0, c in zip(np.cumsum([0] + counts[:-1]), counts)])

        # text units (char offsets in the content string -> template -> final string)
        if isinstance(prompt, DocPrompt):
            off = 0
            for p in prompt.parts:
                if p["type"] == "text":
                    for u in p.get("units", []):
                        pos = positions(base + off + u["start"], base + off + u["end"])
                        if len(pos):
                            uid = tl.add_unit(pos, TEXT, u["label"], sid=u["sid"])
                            hide[uid] = {"text": pos.tolist()}
                    off += len(p["text"]) + 1
                else:
                    off += len(IMAGE_MARK) + 1
        else:
            t_base = base + len(content) - len(prompt.text)
            for u in prompt.units:
                pos = positions(t_base + u["start"], t_base + u["end"])
                if len(pos):
                    uid = tl.add_unit(pos, TEXT, u["label"])
                    hide[uid] = {"text": pos.tolist()}
        meta["timeline"] = tl.summary()
        inputs["_hide"] = hide
        inputs["_tl"] = tl
        return inputs, tl, meta

    def _expand(self, template: str, image_sizes, audios, msn, use_id):
        """The processor's final string (placeholders expanded) and a map template offset -> final offset."""
        ip = self.processor.image_processor
        chunks = re.split(MEDIA_SPLIT, template)
        out, marks, i_img, i_aud, t_pos = [], [], 0, 0, 0
        for ch in chunks:
            rep = ch
            if ch == IMAGE_MARK:
                rep = ip.get_slice_image_placeholder(image_sizes[i_img], i_img, msn, use_id)
                i_img += 1
            elif ch == AUDIO_MARK:       # non-streaming input: one <|audio_start|> ... <|audio_end|> block
                rep = self.processor.get_audio_placeholder(len(audios[i_aud]), chunk_input=False)
                i_aud += 1
            if ch in (IMAGE_MARK, AUDIO_MARK):
                marks.append((t_pos, len(rep) - len(ch)))
            out.append(rep)
            t_pos += len(ch)

        def shift(t: int) -> int:
            return t + sum(d for p, d in marks if p < t)
        return "".join(out), shift

    # ------------------------------------------------------------------ embeddings
    def _data(self, inputs, **override) -> Dict[str, Any]:
        keys = ("input_ids", "pixel_values", "tgt_sizes", "image_bound", "audio_features", "audio_feature_lens",
                "audio_bounds", "spk_bounds")
        d = {k: inputs.get(k) for k in keys}
        d.update(override)
        return d

    def _tok_embed(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.llm.model.embed_tokens(ids)
        return e * self.llm.config.scale_emb if hasattr(self.llm.config, "scale_emb") else e

    @torch.no_grad()
    def prompt_embeds(self, inputs) -> torch.Tensor:
        """[1, N, D] input embeddings exactly as MiniCPMO.generate builds them (cached in `inputs`)."""
        if "_embeds" not in inputs:
            data = self._data(inputs)
            emb, _ = self.model.get_vllm_embedding(data)
            emb = self.model.get_omni_embedding(data, input_embeddings=emb,
                                                chunk_length=self.model.config.audio_chunk_length)
            inputs["_embeds"] = emb
        return inputs["_embeds"]

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, inputs, max_new_tokens: int = 512, recorder=None, force_ids=None) -> Dict[str, Any]:
        """force_ids: emit these tokens instead of greedy ones (AT2 features on a saved response)."""
        emb = self.prompt_embeds(inputs)
        n_prompt = int(emb.shape[1])
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
            out = self.llm.generate(
                inputs_embeds=emb, attention_mask=inputs.get("attention_mask"), max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                eos_token_id=sorted(self.stop_ids), pad_token_id=0, return_dict_in_generate=True, **force)
        finally:
            rec.active = False
            rec.detach(layers)
        seq = out.sequences if hasattr(out, "sequences") else out
        A = rec.stacked()                                   # [T, L, n_prompt]
        gen_ids = seq[0].tolist()
        if len(gen_ids) > A.shape[0]:                       # generation from embeddings returns new tokens only;
            gen_ids = gen_ids[len(gen_ids) - A.shape[0]:]   # guard against versions that prepend the prompt
        T = min(len(gen_ids), A.shape[0])
        return {"gen_ids": gen_ids[:T], "A": A[:T], "gen_seconds": time.time() - t0,
                "n_steps_recorded": int(A.shape[0]), "n_generated": len(gen_ids)}

    # ------------------------------------------------------------------ causal tests
    @torch.no_grad()
    def _hide_cache(self, inputs, fill: str) -> Dict[str, Any]:
        key = f"_hide_{fill}"
        if key in inputs:
            return inputs[key]
        emb = self.prompt_embeds(inputs)[0]
        hide, c = inputs["_hide"], {}
        bounds = inputs["image_bound"][0].tolist() if len(inputs["image_bound"][0]) else []
        c["block_pos"] = [torch.arange(a, b, device=emb.device) for a, b in bounds]
        if bounds:
            vpos = torch.cat(c["block_pos"])
            if fill == "blank":       # every block of a black image, through the vision encoder + resampler
                ip = self.processor.image_processor
                black = (0.0 - ip.mean) / ip.std
                pv = [torch.empty_like(t).copy_(torch.as_tensor(black, dtype=t.dtype).view(-1, 1, 1).expand_as(t))
                      for t in inputs["pixel_values"][0]]
                c["vis_blank"] = self.model.get_vision_embedding(
                    {"pixel_values": [pv], "tgt_sizes": inputs["tgt_sizes"]})[0].to(emb.dtype)   # [blocks, 64, D]
            else:
                c["vis_mean"] = emb[vpos].mean(0)
        if "_apos" in inputs:
            apos = torch.as_tensor(inputs["_apos"], device=emb.device)
            c["apos"] = apos
            if fill == "mean":
                c["aud_mean"] = emb[apos].mean(0)
        text_units = [u for u in inputs["_tl"].units if u["modality"] == "text"]
        pool = [p for u in text_units if u.get("sid") is not None for p in hide[u["uid"]]["text"]] or \
               [p for u in text_units for p in hide[u["uid"]]["text"]]
        if fill == "blank":
            c["text_fill"] = self._tok_embed(torch.tensor([self.space_id], device=emb.device))[0]
        elif pool:
            c["text_fill"] = emb[torch.tensor(pool, device=emb.device)].mean(0)
        inputs[key] = c
        return c

    @torch.no_grad()
    def _blank_audio_embeds(self, inputs, token_sets: Sequence[List[int]]) -> List[torch.Tensor]:
        """Audio embeddings [n_audio, D] with the given audio tokens silenced in the log-mel input (one per set)."""
        feats, lens = inputs["audio_features"], inputs["audio_feature_lens"][0]
        lens = lens.to(feats.device) if torch.is_tensor(lens) else torch.as_tensor(lens, device=feats.device)
        S = feats.shape[0]
        _, tok_per_seg = self.model._get_feat_extract_output_lengths(lens)
        start = torch.cumsum(torch.cat([tok_per_seg.new_zeros(1), tok_per_seg[:-1]]), 0).tolist()
        floor = min(float(feats[s, :, :int(lens[s])].min()) for s in range(S))
        x = feats.repeat(len(token_sets), 1, 1)
        for b, toks in enumerate(token_sets):
            for t in toks:
                s = max(i for i in range(S) if start[i] <= t)
                f0 = (t - start[s]) * MEL_PER_TOKEN
                x[b * S + s, :, f0:min(f0 + MEL_PER_TOKEN, int(lens[s]))] = floor
        out = self.model.get_audio_embedding({"audio_features": x, "audio_feature_lens": [lens] * len(token_sets)},
                                             chunk_length=self.model.config.audio_chunk_length)
        return [torch.cat(o, 0) for o in out]

    @torch.no_grad()
    def hidden_logits(self, inputs, gen_ids: List[int], unit_sets: Sequence[Sequence[int]], fill: str = "blank"):
        """Teacher-forced logits for the response tokens, one batch copy per unit set (uids hidden in that copy).
        Returns [B, T, V]; position t predicts gen_ids[t]."""
        c = self._hide_cache(inputs, fill)
        hide = inputs["_hide"]
        emb = self.prompt_embeds(inputs)[0]
        T, B = len(gen_ids), len(unit_sets)
        gen = self._tok_embed(torch.tensor(gen_ids, device=emb.device))
        x = torch.cat([emb, gen], 0).unsqueeze(0).repeat(B, 1, 1)
        audio_sets = {b: [t for u in us for t in hide[u].get("audio_tokens", [])] for b, us in enumerate(unit_sets)}
        audio_sets = {b: t for b, t in audio_sets.items() if t}
        if audio_sets and fill == "blank":
            new = self._blank_audio_embeds(inputs, list(audio_sets.values()))
            for b, a in zip(audio_sets, new):
                x[b, c["apos"]] = a.to(x.dtype)
        for b, us in enumerate(unit_sets):
            for u in us:
                h = hide[u]
                for blk in h.get("blocks", []):
                    x[b, c["block_pos"][blk]] = c["vis_blank"][blk] if fill == "blank" else c["vis_mean"]
                if "text" in h:
                    x[b, torch.tensor(h["text"], device=x.device)] = c["text_fill"]
                if "audio_tokens" in h and fill != "blank":
                    rows = c["apos"][torch.tensor(h["audio_tokens"], device=x.device)]
                    x[b, rows] = c["aud_mean"] if fill == "mean" else 0
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(B, -1)
        try:
            out = self.llm(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=T + 1)
        except TypeError:
            out = self.llm(inputs_embeds=x, position_ids=pos, use_cache=False)
        L = out.logits
        n = emb.shape[0]
        return L[:, :-1] if L.shape[1] == T + 1 else L[:, n - 1:n - 1 + T]
