"""Generated-text segmentation: response -> sentences (attribution spans), aligned to generated tokens.

Alignment uses the generated token ids themselves (incremental decoding -> character offsets), so
no re-tokenization of the response is needed (v1 re-tokenized and assumed identical ids).
Sentence rules follow v1: spaCy sentences, split again at newlines, bare list markers ("1.", "a.")
merged into the following span.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_NLP = None
_LIST_MARKER = re.compile(r"^\s*(\d+|[a-zA-Z])\.\s*$")


def get_nlp(model: str = "en_core_web_sm"):
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load(model)
    return _NLP


def token_char_offsets(tokenizer, token_ids: List[int]) -> Tuple[str, List[Tuple[int, int]]]:
    """Character span of each generated token in the decoded response (special tokens -> empty)."""
    offsets, prev = [], ""
    for k in range(len(token_ids)):
        cur = tokenizer.decode(token_ids[: k + 1], skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
        if not cur.startswith(prev):          # decoding changed earlier text (rare byte merges)
            prev = prev[: len(cur)]
        offsets.append((len(prev), len(cur)))
        prev = cur
    return prev, offsets


def _sentence_char_spans(text: str) -> List[Tuple[int, int]]:
    doc = get_nlp()(text)
    spans: List[Tuple[int, int]] = []
    for sent in doc.sents:
        start = sent.start_char
        for m in re.finditer(r"\n+", text[sent.start_char: sent.end_char]):
            end = sent.start_char + m.start()
            if text[start:end].strip():
                spans.append((start, end))
            start = sent.start_char + m.end()
        if text[start: sent.end_char].strip():
            spans.append((start, sent.end_char))
    merged: List[Tuple[int, int]] = []
    i = 0
    while i < len(spans):
        s, e = spans[i]
        if _LIST_MARKER.match(text[s:e]) and i + 1 < len(spans):
            merged.append((s, spans[i + 1][1]))
            i += 2
        else:
            merged.append((s, e))
            i += 1
    return merged


def segment_response(text: str, offsets: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """Sentences with the generated-token indices they cover and a POS tag per covered token."""
    doc = get_nlp()(text)
    tok_pos = []
    for (a, b) in offsets:
        tag = "SPACE"
        if b > a:
            best = 0
            for t in doc.char_span(a, b, alignment_mode="expand") or []:
                ov = min(b, t.idx + len(t.text)) - max(a, t.idx)
                if ov > best:
                    best, tag = ov, t.pos_
        tok_pos.append(tag)

    sentences = []
    for (s, e) in _sentence_char_spans(text):
        toks = [k for k, (a, b) in enumerate(offsets) if b > a and a < e and b > s]
        sentences.append({
            "text": text[s:e].strip(),
            "char_span": [s, e],
            "token_idx": toks,
            "pos": [tok_pos[k] for k in toks],
        })
    return sentences
