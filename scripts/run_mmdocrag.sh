#!/usr/bin/env bash
# Image + text: MMDocRAG long-answer QA, 400 questions, each with 15 quotes (text passages and images).
#   bash scripts/run_mmdocrag.sh smoke     # 5 questions on GPU 0, stops at the first error
#   bash scripts/run_mmdocrag.sh full      # generation + attribution for all 400   -> runs/mmdocrag/$MODEL
#   bash scripts/run_mmdocrag.sh causal    # causal test on the finished run (hide top 1-3 quotes vs random), then the summary
# Env: as in run_nextgqa.sh (MODEL, OMNITRACE_DATA, MODEL_PATH, NGPU, BATCH, SETTINGS, FILL = blank (black image /
#      space tokens), TAG, LABELS_ONLY); PER_GPU default 2; IMAGE_MAX_PIXELS (default 1024*28*28 = <= 1024 tokens per image).
# Long answers are graded by an LLM judge, not by this script. Label threshold: configs/label_tau.json (1.5).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-smoke}"
NGPU="${NGPU:-4}"
MODEL="${MODEL:-qwen2.5-omni-7b}"
ROOT="${OMNITRACE_DATA:-data}"
Q="$ROOT/question_image_text.json"
OUT=runs/mmdocrag/$MODEL
EXTRA=()
[[ -n "${MODEL_PATH:-}" ]] && EXTRA+=(--model_path "$MODEL_PATH")
COMMON=(--questions "$Q" --model $MODEL --dump_positions signals --image_max_pixels "${IMAGE_MAX_PIXELS:-802816}" --doc_prompt plain)

case "$MODE" in
  smoke)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u scripts/run_attribution.py "${COMMON[@]}" \
        --out runs/mmdocrag_smoke/$MODEL --limit 5 "${EXTRA[@]}" ;;
  full)
    mkdir -p "$OUT/logs"
    for ((i=0; i<NGPU; i++)); do
      CUDA_VISIBLE_DEVICES=$i nohup python -u scripts/run_attribution.py "${COMMON[@]}" --out "$OUT" \
          --num_shards "$NGPU" --shard_id "$i" "${EXTRA[@]}" > "$OUT/logs/shard$i.log" 2>&1 &
    done
    wait
    echo "done -> $OUT" ;;
  causal)
    mkdir -p "$OUT/logs"
    [[ -n "${BATCH:-}" ]] && EXTRA+=(--batch "$BATCH")
    PER_GPU="${PER_GPU:-2}"; NS=$((NGPU * PER_GPU)); FILL="${FILL:-blank}"
    LO=(); DEFTAG="$FILL"; [[ -n "${LABELS_ONLY:-}" ]] && { LO=(--labels_only); DEFTAG="${FILL}_labels"; }
    TAG="${TAG:-$DEFTAG}"
    read -r -a SET <<< "${SETTINGS:-mass}"
    for ((i=0; i<NS; i++)); do
      CUDA_VISIBLE_DEVICES=$((i % NGPU)) nohup python -u scripts/causal_doc.py run --run "$OUT" --questions "$Q" \
          --split all --settings "${SET[@]}" --fill "$FILL" --tag "$TAG" --num_shards "$NS" --shard_id "$i" \
          "${LO[@]}" "${EXTRA[@]}" > "$OUT/logs/causal_${TAG}_s$i.log" 2>&1 &
    done
    wait
    grep -l "\[error\]\|\[OOM\]" "$OUT"/logs/causal_"${TAG}"_s*.log 2>/dev/null && echo "!! errors / OOM above (rerun the same command: it resumes)" || true
    python scripts/causal_doc.py summarize --run "$OUT" --questions "$Q" --tag "$TAG" --split all | tee "$OUT/logs/causal_${TAG}.log" ;;
  *) echo "unknown mode $MODE (smoke | full | causal)"; exit 1 ;;
esac
