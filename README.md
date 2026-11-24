# CS 289A Grad Project — Self-Consistency as an Unsupervised Post-Training Signal for VQA

Matthew Bernard, Shiqi Yang · Fall 2026

Can self-consistency serve as a reliable unsupervised post-training signal for a
vision-language model on visual question answering? We post-train SmolVLM-500M on its
own high-consensus outputs (no ground-truth labels used for the training signal) and
measure the effect on ScienceQA accuracy.

## Documents
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
   read P(A/B/C/D) after that. Agrees with greedy on 48/50 in a 50-question check.
   The initial read position (last input token, after `"Answer:"`) was found to
   measure a different decision point; the 0.6700 figure from that run was withdrawn.
3. **k = 8 sampled-mode consistency** — next experiment.
4. **Consistency filter** at threshold τ, compared against oracle (ground-truth labels)
   and τ = 0 (no filtering) fine-tuning baselines.
5. **LoRA fine-tuning** of SmolVLM-500M; validation for tuning, test held out.

## Code and results
- `scripts/00_scienceqa_counts.py` — dataset counts per split
- `scripts/01_baseline_eval.py` — zero-shot greedy baseline
- `scripts/02_softmax_eval.py` — corrected softmax-over-options
- `results/baseline_zeroshot.jsonl`, `results/baseline_softmax.jsonl` — per-question outputs

Shiqi's independent checks are on the `shiqi/independent-checks` branch.

## Checkpoints
`huggingface.co/matthew-bernard/cs289-smolvlm-checkpoints`

## Environment
Single TITAN Xp (12 GB), CUDA 12.8, `PreemptMode=REQUEUE` partition; conda `smolvlm`
(Python 3.11), torch 2.14.1+cu126, SmolVLM-500M-Instruct at bf16 (~507.48M params).
Per-example JSONL checkpointing for preemption.

