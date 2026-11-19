#!/bin/bash
# Resumable confirmatory queue (docs/06_protocol.md): each step is skipped when its output exists.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -a; . ./.env; set +a
export PYTHONPATH=. HF_HUB_DISABLE_PROGRESS_BARS=1
log() { echo "$(date -u +%H:%M:%S) $*"; }
run() { out=$1; shift; [ -e "$out" ] && { log "skip $out"; return 0; }; log "run $*"; python -u -W ignore "$@"; }

until [ -e data/synthetic/confirm/messages.parquet ] && ! pgrep -f "scripts.generate_split" >/dev/null; do sleep 60; done
until ! pgrep -f "embed_messages --split confirm" >/dev/null; do sleep 60; done
run data/interim/machine_prob_confirm.parquet -m scripts.machine_confirm
run results/confirm/.done -m scripts.confirm_runs
touch results/confirm/.done
python -u -W ignore -m scripts.analyze_confirm
python -u -W ignore -m scripts.analyze_evasion
log "QUEUE DONE"
