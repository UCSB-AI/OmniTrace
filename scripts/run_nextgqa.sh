#!/usr/bin/env bash
# Video: NExT-GQA (300 test questions, frames only), one shard per GPU.
#   bash scripts/run_nextgqa.sh smoke     # 5 questions on GPU 0, stops at the first error
#   bash scripts/run_nextgqa.sh full      # generation + attribution for all 300   -> runs/nextgqa/$MODEL
#   bash scripts/run_nextgqa.sh causal    # causal test on the finished run (hide top units vs random), then the summary
# Env: MODEL (qwen2.5-omni-7b | minicpm-o-4.5 | qwen3-omni-30b-a3b; each needs its own environment, see README),
#      OMNITRACE_DATA (data root, default ./data), MODEL_PATH (local checkpoint; default: Hugging Face id), NGPU (default 4)
#      causal: PER_GPU (processes per GPU, default 3), BATCH (variants per forward; lower it after [OOM]),
#      SETTINGS (rankings to test, default "mass" = OmniTrace; add v1 for argmax tracing + curation),
#      FILL (blank = black frames, default | mean), TAG (output tag, default = FILL), LABELS_ONLY=1 (only the label rows)
# Label threshold: configs/label_tau.json (1.5).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-smoke}"
NGPU="${NGPU:-4}"
MODEL="${MODEL:-qwen2.5-omni-7b}"
ROOT="${OMNITRACE_DATA:-data}"
Q="$ROOT/question_video.json"
OUT=runs/nextgqa/$MODEL
EXTRA=()
[[ -n "${MODEL_PATH:-}" ]] && EXTRA+=(--model_path "$MODEL_PATH")

case "$MODE" in
  smoke)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u scripts/run_attribution.py --questions "$Q" --model $MODEL \
        --out runs/nextgqa_smoke/$MODEL --limit 5 --dump_positions signals "${EXTRA[@]}" ;;
  full)
    mkdir -p "$OUT/logs"
    for ((i=0; i<NGPU; i++)); do
      CUDA_VISIBLE_DEVICES=$i nohup python -u scripts/run_attribution.py --questions "$Q" --model $MODEL --out "$OUT" \
          --num_shards "$NGPU" --shard_id "$i" --dump_positions signals "${EXTRA[@]}" > "$OUT/logs/shard$i.log" 2>&1 &
    done
    wait
    echo "done -> $OUT" ;;
  causal)
    mkdir -p "$OUT/logs"
    [[ -n "${BATCH:-}" ]] && EXTRA+=(--batch "$BATCH")
    PER_GPU="${PER_GPU:-3}"; NS=$((NGPU * PER_GPU)); FILL="${FILL:-blank}"
    LO=(); DEFTAG="$FILL"; [[ -n "${LABELS_ONLY:-}" ]] && { LO=(--labels_only); DEFTAG="${FILL}_labels"; }
    TAG="${TAG:-$DEFTAG}"
    read -r -a SET <<< "${SETTINGS:-mass}"
    for ((i=0; i<NS; i++)); do
      CUDA_VISIBLE_DEVICES=$((i % NGPU)) nohup python -u scripts/causal_video.py run --run "$OUT" --questions "$Q" --n 300 \
          --settings "${SET[@]}" --fill "$FILL" --tag "$TAG" --num_shards "$NS" --shard_id "$i" \
          "${LO[@]}" "${EXTRA[@]}" > "$OUT/logs/causal_${TAG}_s$i.log" 2>&1 &
    done
    wait
    grep -l "\[error\]\|\[OOM\]" "$OUT"/logs/causal_"${TAG}"_s*.log 2>/dev/null && echo "!! errors / OOM above (rerun the same command: it resumes)" || true
    python scripts/causal_video.py summarize --run "$OUT" --tag "$TAG" --subset all     | tee "$OUT/logs/causal_${TAG}_all.log"
    python scripts/causal_video.py summarize --run "$OUT" --tag "$TAG" --subset correct | tee "$OUT/logs/causal_${TAG}_correct.log" ;;
  *) echo "unknown mode $MODE (smoke | full | causal)"; exit 1 ;;
esac
