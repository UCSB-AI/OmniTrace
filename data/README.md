# Evaluation data

The question files hold the three evaluation sets of the paper (949 questions) in OmniTrace's input format. Media are not redistributed: download them from the original datasets and put each file at the `path` listed in the question file, relative to this folder (or to `$OMNITRACE_DATA`).

| File | Source | Questions | Media | Path pattern |
|---|---|---|---|---|
| `question_video.json` | [NExT-GQA](https://github.com/doc-doc/NExT-GQA) test (videos from [NExT-QA](https://github.com/doc-doc/NExT-QA)) | 300, one per video, 10–90 s | 300 videos | `NExT-GQA/NExTVideo/<dir>/<video_id>.mp4` (NExT-QA's own folder layout) |
| `question_audio.json` | [MMAU](https://github.com/Sakshi113/MMAU) test-mini | 249 whose answer depends on part of the clip | 249 clips | `MMAU/audio/<sub-category>/<id>.wav` (`id` = MMAU question id) |
| `question_image_text.json` | [MMDocRAG](https://huggingface.co/datasets/MMDocIR/MMDocRAG) `evaluation_15` | 400, each with 15 quotes | quote images | `MMDocRAG/images/<image file>` (file names as in MMDocRAG's `images.zip`) |

Each record has the source ids (`native_id`) to join back to the original files.

Human evidence:
- NExT-GQA: time spans in `evidence`.
- MMDocRAG: gold quotes in `evidence`.
- MMAU: no evidence labels (`evidence: null`). Audio is evaluated by the causal test only.

MMDocRAG's `meta` holds the reference answers. Never put `meta` or `evidence` into a prompt.
