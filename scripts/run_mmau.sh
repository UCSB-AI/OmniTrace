#!/usr/bin/env bash
# Audio: MMAU test-mini, the 249 questions whose answer depends on part of the clip (options shuffled per question, seeded).
#   bash scripts/run_mmau.sh smoke     # 5 questions on GPU 0, stops at the first error
#   bash scripts/run_mmau.sh full      # generation + attribution for all 249   -> runs/mmau/$MODEL
#   bash scripts/run_mmau.sh causal    # causal test on the finished run (hide top seconds vs random), then the summary
# Env: as in run_nextgqa.sh (MODEL, OMNITRACE_DATA, MODEL_PATH, NGPU, PER_GPU, BATCH, SETTINGS, FILL = blank (silence) | mean,
#      TAG, LABELS_ONLY). Label threshold: configs/label_tau.json (1.5).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-smoke}"
NGPU="${NGPU:-4}"
MODEL="${MODEL:-qwen2.5-omni-7b}"
ROOT="${OMNITRACE_DATA:-data}"
Q="$ROOT/question_audio.json"
OUT=runs/mmau/$MODEL
EXTRA=()
[[ -n "${MODEL_PATH:-}" ]] && EXTRA+=(--model_path "$MODEL_PATH")
COMMON=(--questions "$Q" --model $MODEL --shuffle_options --seed 0 --dump_positions signals)

case "$MODE" in
  smoke)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u scripts/run_attribution.py "${COMMON[@]}" \
        --out runs/mmau_smoke/$MODEL --limit 5 "${EXTRA[@]}" ;;
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
    PER_GPU="${PER_GPU:-3}"; NS=$((NGPU * PER_GPU)); FILL="${FILL:-blank}"
    LO=(); DEFTAG="$FILL"; [[ -n "${LABELS_ONLY:-}" ]] && { LO=(--labels_only); DEFTAG="${FILL}_labels"; }
    TAG="${TAG:-$DEFTAG}"
    read -r -a SET <<< "${SETTINGS:-mass}"
    for ((i=0; i<NS; i++)); do
      CUDA_VISIBLE_DEVICES=$((i % NGPU)) nohup python -u scripts/causal_audio.py run --run "$OUT" --questions "$Q" \
          --settings "${SET[@]}" --fill "$FILL" --tag "$TAG" --num_shards "$NS" --shard_id "$i" \
          "${LO[@]}" "${EXTRA[@]}" > "$OUT/logs/causal_${TAG}_s$i.log" 2>&1 &
    done
    wait
    grep -l "\[error\]\|\[OOM\]" "$OUT"/logs/causal_"${TAG}"_s*.log 2>/dev/null && echo "!! errors / OOM above (rerun with PER_GPU=1 BATCH=2: it resumes)" || true
    python scripts/causal_audio.py summarize --run "$OUT" --tag "$TAG"               | tee "$OUT/logs/causal_${TAG}_all.log"
    python scripts/causal_audio.py summarize --run "$OUT" --tag "$TAG" --min_units 5 | tee "$OUT/logs/causal_${TAG}_min5.log" ;;
  *) echo "unknown mode $MODE (smoke | full | causal)"; exit 1 ;;
esac
