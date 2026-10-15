"""Memory-light attention capture during generation.

The LLM runs with SDPA (no N x N attention matrix is ever materialized). For every decoder layer
we additionally compute the attention weights of the *last query row only*, which is the row that
produces the next token:
    w_l = mean_heads softmax(q_last @ K^T * scale + mask_last)          -> [K]
One forward call of the LLM = one decoding step (the prefill call produces the 1st generated token),
so step t's vectors are a_t(i) in the paper, restricted to the prompt positions i < n_prompt.

Mechanism: the global "sdpa" entry of transformers' AttentionInterface is wrapped. Only modules that
were explicitly tagged by `AttentionRecorder.attach` are recorded, so vision/audio encoders, which
also use SDPA, pass through untouched. This needs a transformers version in which the model's
decoder attention dispatches through `ALL_ATTENTION_FUNCTIONS` (checked in `attach`).
"""
from __future__ import annotations

import inspect
from typing import List, Optional

import torch

_ORIG_SDPA = None


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _last_row_weights(module, query, key, attention_mask, scaling, per_head: bool = False) -> torch.Tensor:
    """Attention weights of the last query position, float32: head-averaged [K], or [H, K] with per_head."""
    q = query[:, :, -1:, :].float()                                   # [B, H, 1, D]
    k = _repeat_kv(key, query.shape[1] // key.shape[1]).float()       # [B, H, K, D]
    s = torch.matmul(q, k.transpose(-1, -2)) * scaling                # [B, H, 1, K]
    if attention_mask is not None and attention_mask.dim() == 4:
        m = attention_mask[:, :, -1:, : k.shape[2]]
        s = s.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else s + m.float()
    w = torch.softmax(s, dim=-1)[0, :, 0]                             # [H, K]
    return w if per_head else w.mean(0)                               # [K]


def _capturing_sdpa(module, query, key, value, attention_mask, *args, **kwargs):
    out = _ORIG_SDPA(module, query, key, value, attention_mask, *args, **kwargs)
    rec: Optional[AttentionRecorder] = getattr(module, "_ot_recorder", None)
    if rec is not None and rec.active:
        scaling = kwargs.get("scaling")
        if scaling is None:
            scaling = getattr(module, "scaling", query.shape[-1] ** -0.5)
        with torch.no_grad():
            rec.record(module._ot_layer, _last_row_weights(module, query, key, attention_mask, scaling,
                                                           per_head=getattr(rec, "per_head", False)))
    return out


def install() -> None:
    """Wrap the global SDPA attention function (idempotent)."""
    global _ORIG_SDPA
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if _ORIG_SDPA is None:
        _ORIG_SDPA = ALL_ATTENTION_FUNCTIONS["sdpa"]
        try:
            from transformers import AttentionInterface
            AttentionInterface.register("sdpa", _capturing_sdpa)
        except ImportError:                       # older transformers: a plain dict
            ALL_ATTENTION_FUNCTIONS["sdpa"] = _capturing_sdpa


class AttentionRecorder:
    """Collects [n_layers, n_prompt] last-row attention per decoding step (on the model's device)."""

    def __init__(self, n_prompt: int, keep_device: bool = True):
        self.n_prompt = n_prompt
        self.active = False
        self.n_layers = 0
        self.steps: List[torch.Tensor] = []     # each [L, n_prompt] float32
        self._cur: List[Optional[torch.Tensor]] = []

    def attach(self, decoder_layers) -> None:
        """Tag the self-attention module of every decoder layer of the language model."""
        install()
        self.n_layers = len(decoder_layers)
        for i, layer in enumerate(decoder_layers):
            attn = layer.self_attn
            src = inspect.getsource(type(attn).forward)
            if "ALL_ATTENTION_FUNCTIONS" not in src and "attention_interface" not in src:
                raise RuntimeError(
                    f"{type(attn).__name__} does not dispatch through ALL_ATTENTION_FUNCTIONS; "
                    "upgrade transformers (>=4.56) to use SDPA capture.")
            attn._ot_recorder = self
            attn._ot_layer = i
        self._cur = [None] * self.n_layers

    def detach(self, decoder_layers) -> None:
        for layer in decoder_layers:
            layer.self_attn._ot_recorder = None

    def record(self, layer: int, w: torch.Tensor) -> None:
        # Layers may sit on different GPUs (device_map="auto"); keep every row on layer 0's device.
        dev = self._cur[0].device if self._cur[0] is not None else w.device
        self._cur[layer] = w[: self.n_prompt].to(dev, non_blocking=True)
        if layer == self.n_layers - 1:            # last layer closes one forward call = one step
            self.steps.append(torch.stack(self._cur, 0))
            self._cur = [None] * self.n_layers

    def stacked(self) -> torch.Tensor:
        """[T, L, n_prompt] float32 on CPU."""
        if not self.steps:
            return torch.zeros(0, self.n_layers, self.n_prompt)
        return torch.stack(self.steps, 0).cpu()


class ForcedTokens:
    """Logits processor that makes generation emit the given tokens (teacher forcing through generate()).
    Reads attention on a SAVED response on any GPU, where free greedy decoding can drift (bf16 kernels differ).
    Counts its calls (one per step), so it works whether or not input_ids include the prompt."""

    def __init__(self, ids):
        self.ids = [int(x) for x in ids]
        self.step = 0

    def __call__(self, input_ids, scores):
        t = self.ids[min(self.step, len(self.ids) - 1)]
        self.step += 1
        out = torch.full_like(scores, float("-inf"))
        out[:, t] = 0.0
        return out


class HeadUnitRecorder(AttentionRecorder):
    """Per-head variant: every step keeps the attention MASS of each head on each unit, [L, H, U]
    (positions are summed into their unit on the fly, so no [H, n_prompt] rows are stored).
    Used to fit the head weights of OmniTrace-trained; pass it to backend.generate(recorder=...)."""

    per_head = True

    def __init__(self, n_prompt: int, pos_unit, n_units: int):
        super().__init__(n_prompt)
        pu = torch.as_tensor(pos_unit, dtype=torch.long)[:n_prompt]
        self.n_units = n_units
        self._src = pu >= 0
        self._idx = pu[self._src]

    def record(self, layer: int, w: torch.Tensor) -> None:      # w [H, K]
        src, idx = self._src.to(w.device), self._idx.to(w.device)
        u = torch.zeros(w.shape[0], self.n_units, device=w.device).index_add_(1, idx, w[:, : self.n_prompt][:, src])
        dev = self._cur[0].device if self._cur[0] is not None else u.device
        self._cur[layer] = u.to(dev, non_blocking=True)
        if layer == self.n_layers - 1:
            self.steps.append(torch.stack(self._cur, 0))
            self._cur = [None] * self.n_layers

    def stacked(self) -> torch.Tensor:
        """[T, L, H, U] float32 on CPU."""
        return torch.stack(self.steps, 0).cpu() if self.steps else torch.zeros(0, self.n_layers, 0, self.n_units)
