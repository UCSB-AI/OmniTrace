"""Shared v2 question schema: loading, media paths, prompt construction, answer parsing.

Record layout (see data/<DATASET>/build_question_json.py):
    {"id", "dataset", "split", "task": "mcqa", "question", "options", "answer", "answer_idx",
     "sources": [{"sid", "modality": "video"|"audio"|..., "path" (relative to DATA_ROOT), "duration", ...}],
     "evidence": [{"sid", "spans": [[start, end], ...]}] | None, "meta": {...}}
"""
from __future__ import annotations

import json
import os
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("OMNITRACE_DATA", REPO_ROOT / "data"))

LETTERS = string.ascii_uppercase

# Instruction appended after the question/options (video/audio first, then this text, as in v1).
MCQA_INSTRUCTION = (
    "Answer the question based on the {modality} provided. Explain your reasoning step by step. "
    "End your response with a final line of the form \"Answer: X\", where X is the option letter."
)
# Self-citation baseline (P0) for video / audio: the same instruction plus a request to cite time spans
# Separate run; the main runs use MCQA_INSTRUCTION.
MCQA_CITE_INSTRUCTION = (
    "Answer the question based on the {modality} provided. Explain your reasoning step by step. "
    "After each sentence that uses information from the {modality}, cite the time span it comes from in "
    "seconds, e.g. [12.0-15.5s]. "
    "End your response with a final line of the form \"Answer: X\", where X is the option letter."
)

# Long-form QA over quotes (MMDocRAG). "plain" = main setting (no citation request, so the response is
# not shaped by our evaluation); "cite" = MMDocRAG's own citation format (self-citation baseline P0).
DOC_INSTRUCTIONS = {
    "plain": ("Answer the question based on the quotes provided. Write a detailed answer that ties back to "
              "the relevant quotes, then conclude with a direct and concise answer in one sentence."),
    "cite": ("Answer the question based on the quotes provided. Write a detailed answer that ties back to "
             "the relevant quotes, then conclude with a direct and concise answer in one sentence. Cite text "
             "quotes as [index], e.g. [1] for the first text quote, and cite images as ![one-sentence summary]"
             "(image index), e.g. ![...](image1) for the first image. Do not cite irrelevant quotes."),
}


def load_questions(path: str | Path, limit: Optional[int] = None,
                   num_shards: int = 1, shard_id: int = 0,
                   ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    items = json.loads(Path(path).read_text())
    if ids:
        keep = set(ids)
        items = [x for x in items if x["id"] in keep]
    if limit is not None:
        items = items[:limit]
    return items[shard_id::num_shards]


def media_path(source: Dict[str, Any], data_root: Path = DATA_ROOT) -> str:
    p = Path(source["path"])
    return str(p if p.is_absolute() else data_root / p)


@dataclass
class Prompt:
    """User-text part of the input, with labelled character ranges (= text source units)."""
    text: str
    units: List[Dict[str, Any]] = field(default_factory=list)  # {"label", "start", "end"} char offsets

    def add(self, label: str, piece: str, newline: bool = True) -> None:
        start = len(self.text)
        self.text += piece
        self.units.append({"label": label, "start": start, "end": len(self.text)})
        if newline:
            self.text += "\n"


@dataclass
class DocPrompt:
    """Interleaved user message for quote-based QA (MMDocRAG layout: text quotes, then image quotes, then the
    question). `parts` = [{"type": "text", "text", "units": [{"label", "sid", "start", "end"}]} |
    {"type": "image", "sid", "path"}]; unit char offsets are relative to their part. Scaffolding ("[k] ",
    "- imagek is ", headers) belongs to no unit."""
    parts: List[Dict[str, Any]] = field(default_factory=list)

    def text_part(self) -> Dict[str, Any]:
        if not self.parts or self.parts[-1]["type"] != "text":
            self.parts.append({"type": "text", "text": "", "units": []})
        return self.parts[-1]

    def add(self, piece: str, label: Optional[str] = None, sid: Optional[str] = None) -> None:
        part = self.text_part()
        start = len(part["text"])
        part["text"] += piece
        if label is not None:
            part["units"].append({"label": label, "sid": sid, "start": start, "end": len(part["text"])})

    def add_image(self, sid: str, path: str) -> None:
        self.parts.append({"type": "image", "sid": sid, "path": path})

    @property
    def text(self) -> str:
        """Readable prompt for records (images shown as <image:sid>)."""
        return "".join(p["text"] if p["type"] == "text" else f"<image:{p['sid']}>" for p in self.parts)


def build_doc_prompt(sample: Dict[str, Any], style: str = "plain") -> DocPrompt:
    """MMDocRAG's interleaved layout: 'Text Quotes are:' [k] text ... 'Image Quotes are:' - imagek is <img> ...
    'The user question is: ...' + instruction. Units: each quote (label = sid), question, instruction."""
    p = DocPrompt()
    texts = [s for s in sample["sources"] if s["modality"] == "text"]
    images = [s for s in sample["sources"] if s["modality"] == "image"]
    p.add("Text Quotes are:\n")
    for s in texts:
        p.add(f"[{s['sid'][4:]}] ")
        p.add(" ".join(s["text"].split()), label=s["sid"], sid=s["sid"])
        p.add("\n")
    p.add("Image Quotes are:\n")
    for s in images:
        p.add(f"- {s['sid']} is ")
        p.add_image(s["sid"], s["path"])
        p.add("\n")
    p.add("The user question is: ")
    p.add(sample["question"].strip(), label="question")
    p.add("\n")
    p.add(DOC_INSTRUCTIONS[style], label="instruction")
    return p


def shuffle_options(sample: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
    """Copy of an MCQA sample with its options in a seeded per-question order (MMAU: the correct answer
    is option A in 117/249). `option_order[i]` = original index of the option shown at position i;
    `answer_idx` refers to the shown order. Deterministic in (seed, id), so reruns rebuild the same prompt."""
    import random
    order = list(range(len(sample["options"])))
    random.Random(f"{seed}:{sample['id']}").shuffle(order)
    out = dict(sample)
    out["options"] = [sample["options"][i] for i in order]
    if sample.get("answer_idx") is not None:
        out["answer_idx"] = order.index(sample["answer_idx"])
    out["option_order"] = order
    return out


def build_mcqa_prompt(sample: Dict[str, Any], modality: str = "video", style: str = "plain") -> Prompt:
    """question / lettered options / instruction, each line a separate text unit.
    style: plain (main) | cite (self-citation baseline: also asks for cited time spans)."""
    p = Prompt(text="")
    q = sample["question"].strip()
    if q and q[-1] not in "?.!":
        q += "?"
    p.add("question", q[0].upper() + q[1:] if q else q)
    for i, opt in enumerate(sample["options"]):
        p.add(f"option_{LETTERS[i]}", f"{LETTERS[i]}. {opt}")
    instr = MCQA_CITE_INSTRUCTION if style == "cite" else MCQA_INSTRUCTION
    p.add("instruction", instr.format(modality=modality), newline=False)
    return p


# "Answer: C", "answer is (B)", "**Answer:** D", 'the answer is definitely "C".', "I'd say the answer is
# probably "C" for ...": keyword case-insensitive, up to two adverbs, optional quotes/brackets/bold;
# the letter itself must be uppercase.
_ANSWER_RE = re.compile(
    r"(?i:answer)\s*(?:(?i:is|would be|should be|will be)\s+)?(?:[a-z]+ly\s+){0,2}[:：]?\s*"
    r"[\*\"'“‘(\[]*\s*([A-Z])(?![A-Za-z])")


def parse_answer(response: str, options: List[str]) -> Tuple[Optional[int], str]:
    """Return (option index or None, how it was parsed)."""
    n = len(options)
    valid = LETTERS[:n]
    hits = [m.group(1).upper() for m in _ANSWER_RE.finditer(response)]
    hits = [h for h in hits if h in valid]
    if hits:
        return valid.index(hits[-1]), "answer_tag"
    letters = re.findall(r"(?:^|[\s(])([A-Z])[.):](?:\s|$)", response)
    letters = [l for l in letters if l in valid]
    if letters:
        return valid.index(letters[-1]), "letter"
    # option text as whole words; prefer the longest matching option, then the latest mention
    low = response.lower()
    found = []
    for i, o in enumerate(options):
        o = o.strip().lower()
        if len(o) < 3:
            continue
        ms = list(re.finditer(r"\b" + re.escape(o) + r"\b", low))
        if ms:
            found.append((len(o), ms[-1].start(), i))
    if found:
        return max(found)[2], "option_text"
    return None, "none"
