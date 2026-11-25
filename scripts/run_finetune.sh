#!/bin/bash
# Train the three adapters (filtered, oracle, tau=0) and evaluate each on the 500-question test subset.
set -e
cd "$(dirname "$0")"
P=${PYTHON:-python}
for name in filtered_tau0.9 oracle_tau0.9 tau0; do
  $P 05_train_lora.py --train ../data/pseudolabels/$name.jsonl --out ../checkpoints/$name --no-image-splitting
  $P 03_softmax_scores.py --split test --n 500 --no-image-splitting --adapter ../checkpoints/$name \
      --out ../results/scores/${name}_test500.jsonl
done
