#!/bin/bash
# Resumable exploration queue: a step is skipped when its output exists; long steps checkpoint themselves.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -a; . ./.env; set +a
export PYTHONPATH=. HF_HUB_DISABLE_PROGRESS_BARS=1
log() { echo "$(date -u +%H:%M:%S) $*"; }
run() { out=$1; shift; [ -e "$out" ] && { log "skip $out"; return 0; }; log "run $*"; python -u -W ignore "$@"; }

until [ -e data/synthetic/explore/messages.parquet ] && ! pgrep -f "generate_split" >/dev/null; do sleep 60; done
run data/interim/machine_prob_explore.parquet -m scripts.machine_scores
until ! pgrep -f "embed_messages --split explore --name vk_explore_luceri" >/dev/null; do sleep 60; done
run results/explore/day_top3/.done -m scripts.explore_runs --tag day_top3 --gnn --luceri \
    --set null_model.bucket=community_day fusion.account_agg=top3
touch results/explore/day_top3/.done
run results/explore/hour_top3/.done -m scripts.explore_runs --tag hour_top3 \
    --set null_model.bucket=community_hour fusion.account_agg=top3
touch results/explore/hour_top3/.done
run results/explore/day_max/.done -m scripts.explore_runs --tag day_max \
    --set null_model.bucket=community_day fusion.account_agg=max
touch results/explore/day_max/.done
for t in day_top3 hour_top3 day_max; do python -W ignore -m scripts.summarize_explore --tag $t > /dev/null; done
log "QUEUE DONE"
