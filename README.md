# CS 289A Grad Project — Self-Consistency as an Unsupervised Post-Training Signal for VQA

Matthew Bernard, Shiqi Yang · Fall 2026

Can self-consistency serve as a reliable unsupervised post-training signal for a
vision-language model on visual question answering? We post-train SmolVLM-500M on its
own high-consensus outputs (no ground-truth labels used for the training signal) and
measure the effect on ScienceQA accuracy.

## Documents
- First draft, ICML template (with fine-tuning results and figures): `paper/main.tex` / `paper/main.pdf`
- First draft: `grad-project-firstdraft.qmd` / `grad-project-firstdraft.pdf`
- Progress report: `grad-project-progress.qmd` / `grad-project-progress.pdf`
- Outline: `grad-project-outline.qmd` / `grad-project-outline.pdf`
- Proposal: `grad-project-proposal.qmd` / `grad-project-proposal.pdf`

## Data
- Main: `lmms-lab/ScienceQA-IMG` (image-only) — 6,218 / 2,097 / 2,017 train/val/test
- Secondary: text-only rows from `derek-thomas/ScienceQA` (row-aligned with the
  image-only test split)
- VQAv2: stretch goal only

## Approach
1. **Zero-shot greedy baseline** on a 500-question image-only test subset (seed 0):
   **0.6480 (324/500)**, one autoregressive decode (≤ 8 tokens) per question.
2. **Corrected softmax-over-options**: prefill `" Answer:"` in the assistant turn and
   read P(A/B/C/D) after that. On the full 500-question subset: 0.648, agreeing with
   greedy on 482/500.
   The initial read position (last input token, after `"Answer:"`) was found to
   measure a different decision point; the 0.6700 figure from that run was withdrawn.
3. **k = 8 sampled-mode consistency** — not yet run; all fine-tuning uses the softmax confidence.
4. **Consistency filter** at threshold τ, compared against oracle (ground-truth labels)
   and τ = 0 (no filtering) fine-tuning baselines.
5. **LoRA fine-tuning** of SmolVLM-500M; validation for tuning, test held out. First round
   (500-question test subset): filtered τ = 0.9 → 0.648, τ = 0 → 0.658, oracle → 0.772.
   Full tables in `results/RESULTS.md`.

## Code and results
- `scripts/00_scienceqa_counts.py` — dataset counts per split
- `scripts/01_baseline_eval.py` — zero-shot greedy baseline
- `scripts/02_softmax_eval.py` — corrected softmax-over-options
- `results/baseline_zeroshot.jsonl`, `results/baseline_softmax.jsonl` — per-question outputs
- `results/scores/`, `results/train_logs/`, `results/RESULTS.md` — corrected softmax scores,
  LoRA training logs, and the results summary (see Pipeline below)

Shiqi's independent checks are on the `shiqi/independent-checks` branch.

## Checkpoints
`huggingface.co/matthew-bernard/cs289-smolvlm-checkpoints`

## Environment
Single TITAN Xp (12 GB), CUDA 12.8, `PreemptMode=REQUEUE` partition; conda `smolvlm`
(Python 3.11), torch 2.14.1+cu126, SmolVLM-500M-Instruct at bf16 (~507.48M params).
Per-example JSONL checkpointing for preemption.

## Pipeline

    scripts/00_scienceqa_counts.py     # split sizes, image vs text-only
    scripts/01_baseline_eval.py        # zero-shot greedy baseline
    scripts/02_softmax_eval.py         # original softmax (wrong read position, kept for reference)
    scripts/common.py                  # shared model, prompt, and dataset helpers
    scripts/03_softmax_scores.py       # corrected softmax: prefill " Answer:", read P(letter)
    scripts/04_build_pseudolabels.py   # filter by confidence tau; filtered / tau0 / oracle sets
    scripts/05_train_lora.py           # LoRA fine-tuning on a {idx, label} training set

The fine-tuning loop, run from `scripts/` (works on CUDA, Apple MPS, or CPU):

    # 1. score the train and validation splits with the base model
    python 03_softmax_scores.py --split train      --out ../results/scores/base_train.jsonl
    python 03_softmax_scores.py --split validation --out ../results/scores/base_val.jsonl
    # 2. pick tau from the validation quality table, then build the three training sets
    python 04_build_pseudolabels.py --scores ../results/scores/base_val.jsonl
    python 04_build_pseudolabels.py --scores ../results/scores/base_train.jsonl --tau 0.9 \
        --outdir ../data/pseudolabels
    # 3. train one adapter per training set
    python 05_train_lora.py --train ../data/pseudolabels/filtered_tau0.9.jsonl --out ../checkpoints/filtered
    # 4. evaluate an adapter on the test subset
    python 03_softmax_scores.py --split test --n 500 --adapter ../checkpoints/filtered \
        --out ../results/scores/filtered_test500.jsonl

`scripts/run_finetune.sh` runs steps 3 and 4 for all three training sets, and
`scripts/06_make_figures.py` regenerates the paper figures from `results/scores/`.
