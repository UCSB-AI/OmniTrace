"""Model backends. Each backend exposes:
    tokenizer
    prepare(sample, prompt) -> (inputs, Timeline, meta)
    generate(inputs, max_new_tokens) -> {"gen_ids", "A" [T, L, n_prompt], "gen_seconds", ...}
    hidden_logits(inputs, gen_ids, unit_sets, fill) -> [B, T, V]   (optional; causal tests: MiniCPM-o, Qwen3-Omni.
        Without it the causal scripts hide units through Qwen2.5-Omni's thinker inputs and encoder hooks)
"""
BACKENDS = {
    "qwen2.5-omni-7b": ("qwen25_omni", "Qwen25OmniBackend", "Qwen/Qwen2.5-Omni-7B"),
    "qwen2.5-omni-3b": ("qwen25_omni", "Qwen25OmniBackend", "Qwen/Qwen2.5-Omni-3B"),
    # transformers >= 4.57 (the Qwen env after `pip install -U "transformers>=4.57"`), qwen-omni-utils >= 0.0.9
    "qwen3-omni-30b-a3b": ("qwen3_omni", "Qwen3OmniBackend", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
    # separate environment: requirements_minicpm.txt (transformers==4.51.0)
    "minicpm-o-4.5": ("minicpm_o", "MiniCPMOBackend", "openbmb/MiniCPM-o-4_5"),
}


def get_backend(name: str, model_path: str | None = None, **kwargs):
    import importlib
    module, cls, default_path = BACKENDS[name]
    mod = importlib.import_module(f"{__name__}.{module}")
    return getattr(mod, cls)(model_path or default_path, **kwargs)
