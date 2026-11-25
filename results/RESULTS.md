# Results log

All numbers use the same 500-question seed-0 subset of the `lmms-lab/ScienceQA-IMG` test split
unless noted. Corrected softmax = prefill " Answer:" in the assistant turn, read P(letter) at the
next position (`scripts/03_softmax_scores.py`). Run on a MacBook (Apple MPS, bf16).

## Corrected softmax vs. greedy (Nov 23)

| | Image splitting on | Image splitting off |
|---|---|---|
| Accuracy | 0.648 (324/500) | 0.648 (324/500) |
| Mean option mass | 0.9996 | 1.0000 |
| Agreement with greedy (`baseline_zeroshot.jsonl`) | 482/500 (96.4%) | — |
| Speed | 2.1 s/question | 0.21 s/question |
| Output | `scores/base_test500_split_on.jsonl` | `scores/base_test500.jsonl` |

- With splitting on, softmax and greedy are each right on 8 questions where the other is wrong
  (exact McNemar p = 1). The corrected softmax agrees with the original `baseline_softmax.jsonl`
  on only 379/500, which confirms the original read position was the source of the disagreement.
- Turning image splitting off leaves accuracy unchanged (the two settings disagree on 71
  questions, 29 right each way) and is 10x faster, so the fine-tuning experiments use splitting off.

## Confidence vs. accuracy (test subset, diagnostic only)

Splitting on, by confidence bin:

| Confidence | n | Accuracy |
|---|---:|---:|
| < 0.5 | 10 | 0.200 |
| 0.5–0.75 | 72 | 0.431 |
| 0.75–0.9 | 52 | 0.462 |
| 0.9–0.99 | 85 | 0.506 |
| >= 0.99 | 281 | 0.797 |

Keeping questions with confidence >= tau:

| tau | Kept (on) | Acc (on) | Kept (off) | Acc (off) |
|---:|---:|---:|---:|---:|
| 0 | 500 | 0.648 | 500 | 0.648 |
| 0.7 | 430 | 0.693 | 399 | 0.707 |
| 0.9 | 366 | 0.730 | 320 | 0.747 |
| 0.99 | 281 | 0.797 | 231 | 0.814 |

These use test labels and are only a diagnostic; tau is chosen on the validation split.

## Base-model pseudo-label quality on validation and train (image splitting off)

Base accuracy: validation 0.6423 (2,097 questions), train 0.6702 (6,218 questions).
Outputs: `scores/base_val.jsonl`, `scores/base_train.jsonl`.

| tau | Val kept | Val coverage | Val pseudo-label acc | Train kept | Train pseudo-label acc |
|---:|---:|---:|---:|---:|---:|
| 0 | 2097 | 1.000 | 0.642 | 6218 | 0.670 |
| 0.5 | 1993 | 0.950 | 0.660 | 5972 | 0.685 |
| 0.7 | 1703 | 0.812 | 0.699 | 5068 | 0.726 |
| 0.8 | 1554 | 0.741 | 0.717 | 4565 | 0.752 |
| 0.9 | 1354 | 0.646 | 0.743 | 3985 | 0.780 |
| 0.95 | 1217 | 0.580 | 0.758 | 3555 | 0.807 |
| 0.99 | 933 | 0.445 | 0.812 | 2794 | 0.847 |

Chosen tau = 0.9 for the first runs: on validation it keeps about two-thirds of the questions
and raises pseudo-label accuracy by 10 points. tau = 0.99 is a natural ablation.

Training sets (`data/pseudolabels/`, not committed): `filtered_tau0.9` (3,985 questions,
pseudo-labels), `oracle_tau0.9` (same 3,985 questions, gold labels), `tau0` (all 6,218, pseudo-labels).

## First fine-tuning round (Nov 23)

LoRA on the language model's attention and MLP projections (r=8, alpha=16, dropout 0.05),
AdamW lr 2e-4 with linear decay, effective batch 8, 1 epoch, image splitting off, Apple MPS bf16
(`scripts/run_finetune.sh`). Each run took 28–44 minutes. Evaluated with the corrected softmax on
the 500-question test subset; "fixed / broken" counts questions that flipped from wrong to right /
right to wrong relative to the base model, and p is an exact McNemar test.

| Model | Train examples | Test acc | Fixed / broken | p | Mean conf | Conf of wrong answers | Conf >= 0.99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base (zero-shot) | — | 0.648 | — | — | 0.872 | 0.798 | 231 |
| Filtered, tau = 0.9 | 3,985 | 0.648 | 13 / 13 | 1.00 | 0.956 | 0.925 | 372 |
| No filtering, tau = 0 | 6,218 | 0.658 | 14 / 9 | 0.40 | 0.954 | 0.931 | 374 |
| Oracle (same 3,985, gold labels) | 3,985 | 0.772 | 81 / 19 | < 0.001 | 0.838 | 0.665 | 199 |

Observations:
- Self-training on the model's own pseudo-labels does not change test accuracy, with or without
  filtering. The training loss on the filtered set starts near 0.004 and ends near 5e-5, because
  the model already assigns these labels almost all of its probability.
- Both pseudo-label runs make the model more confident, including on answers that are wrong
  (mean confidence of wrong answers 0.80 -> 0.93). This is the confirmation-bias mechanism behind
  error propagation: a second round would admit more wrong labels past the same threshold.
- The same training on the same questions with gold labels gains 12.4 points (p < 0.001) and
  lowers confidence on wrong answers. The training setup works; the missing ingredient is the
  roughly 22% of pseudo-labels that are wrong and cannot be corrected without outside signal.
