#!/usr/bin/env bash
# overnight_chain.sh — wait for synesthetic-ai chain, then run exp9 attempt #3 Run A + eval
set -u
cd "C:/Users/skell/Desktop/synesthetic-ai"
echo "[watcher] waiting for synesthetic-ai chain (ALL-DONE marker)"
for i in $(seq 1 240); do
  if grep -q ALL-DONE results_probe.log 2>/dev/null; then
    echo "[watcher] synesthetic-ai chain finished at $(date)"
    break
  fi
  sleep 120
done
if ! grep -q ALL-DONE results_probe.log 2>/dev/null; then
  echo "[watcher] gave up after 8h without ALL-DONE marker — aborting, exp9 NOT started"
  exit 1
fi
cd "C:/Users/skell/Desktop/embedding-vibes/experiments/exp9_aspace_encoder"
echo "[watcher] starting exp9 attempt #3 Run A (60k corpus) at $(date)"
py -3.12 train.py --run A --epochs 3 --device cuda > results/train_runA.log 2>&1
TRAINCODE=$?
echo "[watcher] train.py exit code: $TRAINCODE at $(date)"
if [ -f results/ckpt_runA.pt ]; then
  echo "[watcher] starting eval Run A"
  py -3.12 eval.py --run A --device cuda > results/eval_runA.log 2>&1
  echo "[watcher] eval.py exit code: $? at $(date)"
else
  echo "[watcher] no checkpoint produced — skipping eval"
fi
echo "[watcher] EXP9-CHAIN-DONE at $(date)"