"""High-level API: generate an answer and attribute every sentence in one call.

    from omnitrace import OmniTrace
    ot = OmniTrace("qwen2.5-omni-7b")
    out = ot.trace({"question": "...", "options": [...], "sources": [{"sid": "v0", "modality": "video", "path": "clip.mp4"}]})
    for s in out["sentences"]:
        print(s["text"], s["spans"] or s["sources"])

This is the same computation as `scripts/run_attribution.py` with the default `--signals attention`
(late-layer attention mass per source unit, labels by the per-modality threshold in configs/label_tau.json),
without writing records or traces.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

from .backends import BACKENDS, get_backend
from .data import Prompt, build_doc_prompt, build_mcqa_prompt, parse_answer
from .evidence import attention_attribution, modality_tau, run_modality
from .segment import segment_response, token_char_offsets
from .tracing import unit_aggregates

OPEN_INSTRUCTION = "Answer the question based on the {modality} provided. Explain your reasoning step by step."


def _open_prompt(sample: Dict[str, Any], modality: str) -> Prompt:
    """Open-ended question about a video or audio clip (no options)."""
    p = Prompt(text="")
    p.add("question", sample["question"].strip())
    p.add("instruction", OPEN_INSTRUCTION.format(modality=modality), newline=False)
    return p


class OmniTrace:
    """Wraps one model backend. `model`: one of BACKENDS (qwen2.5-omni-7b, qwen2.5-omni-3b, minicpm-o-4.5,
    qwen3-omni-30b-a3b; MiniCPM-o and Qwen3-Omni need their own environments, see the README)."""

    def __init__(self, model: str = "qwen2.5-omni-7b", model_path: Optional[str] = None,
                 label_tau: Optional[float] = None, max_new_tokens: int = 512,
                 video_fps: float = 2.0, video_total_pixels: int = 20480 * 28 * 28,
                 image_max_pixels: int = 1024 * 28 * 28, audio_unit_sec: float = 1.0, **backend_kwargs):
        if model not in BACKENDS:
            raise ValueError(f"unknown model {model!r}; choose from {sorted(BACKENDS)}")
        self.model = model
        self.label_tau = label_tau
        self.max_new_tokens = max_new_tokens
        self.backend = get_backend(model, model_path, video_fps=video_fps, video_total_pixels=video_total_pixels,
                                   image_max_pixels=image_max_pixels, audio_unit_sec=audio_unit_sec, **backend_kwargs)

    def trace(self, sample: Dict[str, Any], data_root: Optional[str] = None) -> Dict[str, Any]:
        """sample: the input schema of the question files (README, "Input format"). Required: `question` and
        `sources`; `options` for multiple choice; `task: "long_qa"` for quote-based document QA.
        Relative media paths resolve against `data_root`; without it, against the working directory when the
        file exists there, else against $OMNITRACE_DATA (default ./data).

        Returns {"response", "pred_idx" (multiple choice), "units", "sentences": [{text, scores, units, spans,
        sources}], "response_attribution": {...}, "label_tau"}. `scores` are indexed by unit id (`units[i]["uid"]`);
        `spans` are merged time intervals per source, `sources` are the labelled quote / image ids."""
        sample = copy.deepcopy(sample)
        sample.setdefault("id", "sample")
        for s in sample["sources"]:
            if "path" in s and not Path(s["path"]).is_absolute():
                if data_root is not None:
                    s["path"] = str((Path(data_root) / s["path"]).resolve())
                elif Path(s["path"]).exists():
                    s["path"] = str(Path(s["path"]).resolve())
        long_qa = sample.get("task") == "long_qa"
        modality = sample["sources"][0]["modality"]
        if long_qa:
            prompt = build_doc_prompt(sample)
        elif sample.get("options"):
            prompt = build_mcqa_prompt(sample, modality=modality)
        else:
            prompt = _open_prompt(sample, modality)

        inputs, tl, _ = self.backend.prepare(sample, prompt)
        gen = self.backend.generate(inputs, max_new_tokens=self.max_new_tokens)
        response, offsets = token_char_offsets(self.backend.tokenizer, gen["gen_ids"])
        sentences = segment_response(response, offsets)
        tau = self.label_tau if self.label_tau is not None else modality_tau(run_modality(tl.units))
        att = attention_attribution(unit_aggregates(gen["A"], tl)["unit_sum"], sentences, tl.units, tau)
        out = {"response": response, "units": tl.units, "sentences": att["sentences"],
               "response_attribution": att["response"], "label_tau": tau}
        if sample.get("options") and not long_qa:
            out["pred_idx"] = parse_answer(response, sample["options"])[0]
        return out
