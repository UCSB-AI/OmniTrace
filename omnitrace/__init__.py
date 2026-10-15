"""OmniTrace: single-pass, training-free source attribution for omni-modal LLMs.

Modules:
  api        OmniTrace: generate + attribute one input in one call
  data       question schema, prompts, answer parsing
  timeline   input position -> (modality, unit); unit -> label / time interval
  capture    attention capture during generation (last query row, SDPA)
  evidence   attention mass per unit (`attention_attribution`), evidence labels, settings on saved traces
  tracing    per-unit aggregates; per-token argmax tracing + curation (the earlier variant, kept for comparison)
  segment    response -> sentences aligned to generated tokens
  ablation   content-free placeholders for the causal test (black frames, silence, blank tokens)
  external   plug-in interface: rank units with your own method and score it with the causal test
  metrics    agreement with human evidence
  backends/  model wrappers: qwen25_omni (Qwen2.5-Omni), minicpm_o (MiniCPM-o 4.5), qwen3_omni (Qwen3-Omni)
"""
__version__ = "0.2.0"


def __getattr__(name):          # lazy, so the CPU-only scripts do not import torch
    if name == "OmniTrace":
        from .api import OmniTrace
        return OmniTrace
    raise AttributeError(name)
