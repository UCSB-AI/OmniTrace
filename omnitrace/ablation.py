"""Hiding source units for the causal tests with an empty placeholder of the same length (input level).

A hidden unit keeps its positions, token count and time ids; only its content is emptied:
    video frame group / image quote  -> black pixels            (blank_patches)
    audio time unit                  -> silence in the log-mel  (blank_audio)
    text quote                       -> space tokens            (blank_text)
The encoders then see a content-free input, so nothing of the hidden unit leaks through neighbouring
tokens, and the LLM sees an input it could meet in real data (unlike a mean or zero embedding).
All functions edit a batch of B copies in place (each copy hides its own units).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch

AUDIO_FRAMES_PER_TOKEN = 4          # 100 mel frames/s -> 25 audio tokens/s (conv stride 2, avg-pool 2)


def black_row(image_processor, dim: int) -> torch.Tensor:
    """One flattened patch row [dim] of a black image after normalisation (rows are channel-major)."""
    mean = torch.tensor(image_processor.image_mean, dtype=torch.float32)
    std = torch.tensor(image_processor.image_std, dtype=torch.float32)
    return (-mean / std).repeat_interleave(dim // len(mean))


def blank_patches(pixels: torch.Tensor, rows: Sequence[Optional[List[int]]], black: torch.Tensor) -> None:
    """pixels [B * n, dim] = B copies back to back; rows[b] = patch rows (within one copy) to blacken."""
    n = pixels.shape[0] // len(rows)
    black = black.to(pixels.device, pixels.dtype)
    for b, r in enumerate(rows):
        if r:
            pixels[b * n + torch.as_tensor(r, dtype=torch.long, device=pixels.device)] = black


def blank_audio(features: torch.Tensor, tokens: Sequence[Optional[List[int]]]) -> None:
    """features [B, n_mels, frames]; tokens[b] = audio-token indices to silence in copy b. Silence is the
    floor of the clip's log-mel features (the value the padding after the clip has)."""
    floor, n_frames = features.min(), features.shape[-1]
    for b, toks in enumerate(tokens):
        if toks:
            fr = [f for k in toks for f in range(k * AUDIO_FRAMES_PER_TOKEN, (k + 1) * AUDIO_FRAMES_PER_TOKEN)
                  if f < n_frames]
            features[b][:, torch.as_tensor(fr, dtype=torch.long, device=features.device)] = floor


def blank_text(input_ids: torch.Tensor, positions: Sequence[Optional[List[int]]], token_id: int) -> None:
    """input_ids [B, N]; positions[b] = prompt positions to overwrite with the placeholder token in copy b."""
    for b, pos in enumerate(positions):
        if pos:
            input_ids[b, torch.as_tensor(pos, dtype=torch.long, device=input_ids.device)] = token_id


def space_token_id(tokenizer) -> int:
    ids = tokenizer.encode(" ", add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"expected one token for a space, got {ids}")
    return ids[0]
