"""Qwen2.5-Omni backend (thinker only, SDPA + last-row attention capture).

Video handling (use_audio_in_video=False, i.e. visual frames only):
  qwen_omni_utils samples frames uniformly at `fps` (default 2) -> the vision encoder merges
  2 consecutive frames (temporal_patch_size=2) and 2x2 patches into one LLM token, so the video
  becomes grid_t temporal groups x (grid_h*grid_w/4) tokens, laid out time-major. Group g covers
  [g * spg, (g+1) * spg] seconds, spg = video_second_per_grid (= 2 / sample_fps). These are the
  same times the model encodes in its TMRoPE temporal position ids (g * spg * 25 ticks).
Each temporal group is one time unit of the timeline.

Audio handling (audio sources, e.g. MMAU):
  qwen_omni_utils loads the wav at 16 kHz -> Whisper mel (100 frames/s, no truncation: the processor
  pads to 300 s) -> the audio encoder (conv stride 2, avg-pool stride 2) emits 25 tokens/s, i.e. audio
  token k covers [k * 0.04, (k + 1) * 0.04] s, the same 40 ms step as its TMRoPE time ids.
  Time units = consecutive runs of round(audio_unit_sec * 25) tokens (default 1 s = 25 tokens); the
  last unit is clipped to the clip duration.

Video with its audio track (audio_in_video=True; NExT-GQA audio-vs-visual reliance run):
  the sound track is decoded here (ffmpeg, else PyAV; 16 kHz mono, the whole clip; qwen_omni_utils' own loader hands
  librosa an audioread object that recent librosa rejects) and passed to the processor; the processor interleaves 2-s chunks (video tokens of the chunk, then
  its audio tokens) between <|vision_bos|><|audio_bos|> ... <|audio_eos|><|vision_eos|>. Both streams stay in time
  order, so the video units are built as above and the audio units as for audio sources (25 tokens = 1 s), with the
  video's sid and the same time axis; the two kinds are told apart by `modality`.

Image+text handling (quote-based QA, e.g. MMDocRAG; prompt = omnitrace.data.DocPrompt):
  the user message interleaves text pieces and images; each image becomes t*h*w/4 <|IMAGE|> tokens
  (image_grid_thw, 2x2 merge), capped by `image_max_pixels`. Units: one IMAGE unit per image quote and one
  TEXT unit per text quote / question / instruction (label = quote sid). Text units are aligned exactly:
  the chat template string is tokenized with offsets, each <|IMAGE|> placeholder is expanded to its token
  count, the result must equal the processor's input_ids, and every unit's character range is located in
  the template.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..capture import AttentionRecorder
from ..data import DocPrompt, Prompt, media_path
from ..timeline import AUDIO, IMAGE, OTHER, TEXT, VIDEO, Timeline

DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-7B"
# Neutral system prompt (decided 2026-10-09). Qwen's official "virtual human" prompt is only needed
# for speech output and made the model chatty and skip the answer format in the smoke test.
SYSTEM_PROMPT = "You are a helpful assistant."
AUDIO_TOKENS_PER_SEC = 25          # 16 kHz -> 100 mel frames/s -> /2 conv -> /2 pool


def load_video_audio(path: str, sr: int = 16000) -> np.ndarray:
    """The sound track of a video file as float32 mono at `sr` (ffmpeg, else PyAV). Raises if there is none."""
    import shutil
    import subprocess
    wav = None
    if shutil.which("ffmpeg"):
        p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                           capture_output=True)
        if p.returncode == 0:
            wav = np.frombuffer(p.stdout, np.float32).copy()
    if wav is None:
        import av
        with av.open(path) as c:
            if not c.streams.audio:
                raise RuntimeError(f"no audio track in {path}")
            rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
            chunks = [f.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for f in rs.resample(fr)]
            chunks += [f.to_ndarray().reshape(-1) for f in rs.resample(None)]
            wav = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    if len(wav) < sr // 10:
        raise RuntimeError(f"no audio track in {path}")
    return wav


class Qwen25OmniBackend:
    name = "qwen2.5-omni"
    audio_in_video = False

    def __init__(self, model_path: str = DEFAULT_MODEL, device_map: str = "auto",
                 video_fps: float = 2.0, video_total_pixels: Optional[int] = None,
                 video_max_pixels: Optional[int] = None, audio_unit_sec: float = 1.0,
                 image_max_pixels: Optional[int] = 1024 * 28 * 28, audio_in_video: bool = False):
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.model_path = model_path
        self.video_fps = video_fps
        self.video_total_pixels = video_total_pixels
        self.video_max_pixels = video_max_pixels
        self.audio_unit_tokens = max(1, int(round(audio_unit_sec * AUDIO_TOKENS_PER_SEC)))
        self.image_max_pixels = image_max_pixels
        self.audio_in_video = audio_in_video
        try:
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map=device_map, attn_implementation="sdpa")
        except TypeError:   # older transformers: torch_dtype
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=device_map, attn_implementation="sdpa")
        if hasattr(model, "disable_talker"):
            model.disable_talker()
        model.eval()
        self.model = model
        self.thinker = model.thinker
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.video_token_id = self.tokenizer.convert_tokens_to_ids("<|VIDEO|>")
        self.vision_eos_id = self.tokenizer.convert_tokens_to_ids("<|vision_eos|>")
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.audio_eos_id = self.tokenizer.convert_tokens_to_ids("<|audio_eos|>")
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|IMAGE|>")
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.stop_ids = {self.im_end_id, self.tokenizer.convert_tokens_to_ids("<|endoftext|>")}

    # ------------------------------------------------------------------ inputs
    def decoder_layers(self):
        return self.thinker.model.layers

    def build_messages(self, sample: Dict[str, Any], prompt: Prompt) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if isinstance(prompt, DocPrompt):
            for part in prompt.parts:
                if part["type"] == "text":
                    content.append({"type": "text", "text": part["text"]})
                else:
                    item = {"type": "image", "image": media_path(part)}
                    if self.image_max_pixels:
                        item["max_pixels"] = self.image_max_pixels
                    content.append(item)
            return [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ]
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
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]

    def prepare(self, sample: Dict[str, Any], prompt: Prompt) -> Tuple[Dict[str, Any], Timeline, Dict[str, Any]]:
        from qwen_omni_utils import process_mm_info

        messages = self.build_messages(sample, prompt)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos, video_kwargs = process_mm_info(messages, use_audio_in_video=False, return_video_kwargs=True)
        if self.audio_in_video:      # the video's own sound track, decoded here (see the module docstring)
            audios = [load_video_audio(media_path(s)) for s in sample["sources"] if s["modality"] == "video"]
        fps_list = video_kwargs.get("fps")
        kwargs = dict(text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
                      padding=True, use_audio_in_video=self.audio_in_video)
        # The processor needs the true sampling fps to set video_second_per_grid (TMRoPE time).
        # Newer transformers take a scalar, older ones a per-video list.
        inputs = None
        fps_args = (([fps_list[0]] if fps_list and len(fps_list) == 1 else []) + [fps_list]) if videos else []
        for fps_arg in fps_args:
            try:
                inputs = self.processor(**kwargs, fps=fps_arg)
                break
            except Exception:
                continue
        if inputs is None:
            inputs = self.processor(**kwargs)
        inputs = inputs.to(self.thinker.device)
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(self.thinker.dtype)

        if isinstance(prompt, DocPrompt):
            tl, meta = self.build_doc_timeline(inputs, prompt, text)
        else:
            tl, meta = self.build_timeline(inputs, sample, prompt, fps_list)
        return inputs, tl, meta

    def build_doc_timeline(self, inputs, prompt: DocPrompt, template: str) -> Tuple[Timeline, Dict[str, Any]]:
        """Image units (one per image) + text units, aligned through the chat-template string."""
        ids = inputs["input_ids"][0].tolist()
        n = len(ids)
        tl = Timeline.empty(n)
        merge = self.thinker.config.vision_config.spatial_merge_size
        grids = inputs["image_grid_thw"].tolist() if inputs.get("image_grid_thw") is not None else []
        counts = [int(t * h * w) // (merge * merge) for t, h, w in grids]

        # template tokens with offsets; expand each <|IMAGE|> placeholder to its token count
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

        # image units, in prompt order
        img_parts = [p for p in prompt.parts if p["type"] == "image"]
        ipos = np.array([i for i, x in enumerate(ids) if x == self.image_token_id], dtype=np.int64)
        assert len(img_parts) == len(counts) and sum(counts) == len(ipos), (len(img_parts), counts, len(ipos))
        a = 0
        for part, c in zip(img_parts, counts):
            tl.add_unit(ipos[a:a + c], IMAGE, part["sid"], sid=part["sid"])
            a += c

        # text units: locate the user content in the template, then each unit's char range
        markup = "<|vision_bos|><|IMAGE|><|vision_eos|>"
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
                    tl.add_unit(np.array(pos), TEXT, u["label"], sid=u["sid"])
        meta = {"n_prompt_tokens": n, "image_grid_thw": grids, "image_tokens": counts,
                "text_units_aligned": True, "timeline": tl.summary()}
        return tl, meta

    def build_timeline(self, inputs, sample, prompt: Prompt, fps_list) -> Tuple[Timeline, Dict[str, Any]]:
        ids = inputs["input_ids"][0].tolist()
        n = len(ids)
        tl = Timeline.empty(n)
        meta: Dict[str, Any] = {"n_prompt_tokens": n}

        # ---- video time units (one per temporal group)
        vpos = np.array([i for i, x in enumerate(ids) if x == self.video_token_id], dtype=np.int64)
        if len(vpos):
            src = next(s for s in sample["sources"] if s["modality"] == "video")
            gt_, gh, gw = [int(x) for x in inputs["video_grid_thw"][0].tolist()]
            merge = self.thinker.config.vision_config.spatial_merge_size
            per_group = gh * gw // (merge * merge)
            assert per_group * gt_ == len(vpos), (per_group, gt_, len(vpos))
            spg = None
            if inputs.get("video_second_per_grid") is not None:
                spg = float(inputs["video_second_per_grid"][0])
            sample_fps = float(fps_list[0]) if fps_list else self.video_fps
            if spg is None:
                spg = 2.0 / sample_fps
            duration = float(src.get("duration") or gt_ * spg)
            for g in range(gt_):
                t0 = g * spg
                t1 = min((g + 1) * spg, duration) if g == gt_ - 1 else (g + 1) * spg
                tl.add_unit(vpos[g * per_group:(g + 1) * per_group], VIDEO, f"t{g}", sid=src["sid"],
                            t0=round(t0, 3), t1=round(max(t1, t0), 3))
            meta.update(video_grid_thw=[gt_, gh, gw], tokens_per_group=per_group, second_per_grid=spg,
                        sample_fps=sample_fps, n_frames=gt_ * 2, video_duration=duration)

        # ---- audio time units (consecutive runs of audio_unit_tokens 40-ms tokens)
        apos = np.array([i for i, x in enumerate(ids) if x == self.audio_token_id], dtype=np.int64)
        if len(apos):
            # an audio source, or the video's own sound track (audio_in_video)
            src = next(s for s in sample["sources"] if s["modality"] == ("video" if self.audio_in_video else "audio"))
            step = 1.0 / AUDIO_TOKENS_PER_SEC
            duration = float(src.get("duration") or len(apos) * step)
            k = self.audio_unit_tokens
            for g, a in enumerate(range(0, len(apos), k)):
                b = min(a + k, len(apos))
                t0, t1 = a * step, min(b * step, duration) if b == len(apos) else b * step
                tl.add_unit(apos[a:b], AUDIO, f"t{g}", sid=src["sid"], t0=round(t0, 3), t1=round(max(t1, t0), 3))
            meta.update(n_audio_tokens=int(len(apos)), audio_unit_tokens=k, audio_unit_sec=k * step,
                        audio_duration=duration, audio_in_video=self.audio_in_video)

        # ---- text units: user prompt tokens between the last media end tag and <|im_end|>
        ends = [i for i, x in enumerate(ids) if x in (self.vision_eos_id, self.audio_eos_id)]
        start = max(ends) + 1 if ends else None
        if start is not None:
            end = next(i for i in range(start, n) if ids[i] == self.im_end_id)
            enc = self.tokenizer(prompt.text, add_special_tokens=False, return_offsets_mapping=True)
            if enc["input_ids"] == ids[start:end]:
                offs = enc["offset_mapping"]
                for u in prompt.units:
                    pos = [start + k for k, (a, b) in enumerate(offs)
                           if b > a and a < u["end"] and b > u["start"] and prompt.text[a:b].strip()]
                    if pos:
                        tl.add_unit(np.array(pos), TEXT, u["label"])
                meta["text_units_aligned"] = True
            else:
                tl.add_unit(np.arange(start, end), TEXT, "prompt")
                meta["text_units_aligned"] = False
        meta["timeline"] = tl.summary()
        return tl, meta

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
        # Text-only generation straight from the thinker. This is what
        # Qwen2_5OmniForConditionalGeneration.generate(..., return_audio=False) does internally, but
        # that wrapper touches self.talker, which disable_talker() deleted. Stop tokens are set
        # explicitly so we do not rely on the thinker's generation_config.
        try:
            out = self.thinker.generate(
                **inputs, use_audio_in_video=self.audio_in_video, max_new_tokens=max_new_tokens, do_sample=False,
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
