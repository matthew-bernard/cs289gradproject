"""Make the paper figures from the score files in results/scores/.

  fig_tau.pdf         pseudo-label accuracy vs. fraction of questions kept, one point per tau
  fig_finetune.pdf    test accuracy of the base model and the three fine-tuned adapters
  fig_confidence.pdf  confidence on wrongly answered test questions, before and after fine-tuning
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
SCORES = os.path.join(ROOT, "results", "scores")
OUT = os.path.join(ROOT, "paper", "figures")
TAUS = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42,
})


def load(name):
    with open(os.path.join(SCORES, name)) as f:
        return [json.loads(line) for line in f]


def tau_curve(rows):
    pts = []
    for t in TAUS:
        kept = [r for r in rows if r["conf"] >= t]
        pts.append((len(kept) / len(rows), sum(r["ok"] for r in kept) / len(kept)))
    return pts


def fig_tau():
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    for name, color, marker, label in [("base_val.jsonl", BLUE, "o", "Validation"),
                                       ("base_train.jsonl", ORANGE, "s", "Train")]:
        pts = tau_curve(load(name))
        xs, ys = zip(*pts)
        ax.plot(xs, [100 * y for y in ys], color=color, lw=2, marker=marker, ms=4,
                markeredgecolor="white", markeredgewidth=0.8, label=label)
        if label == "Validation":
            for t, (x, y) in zip(TAUS, pts):
                ax.annotate(f"$\\tau$={t:g}", (x, 100 * y), textcoords="offset points",
                            xytext=(0, -11), ha="center", fontsize=6, color=MUTED)
    ax.set_xlabel("Fraction of questions kept")
    ax.set_ylabel("Pseudo-label accuracy (%)")
    ax.set_xlim(1.03, 0.4)
    ax.set_ylim(58, 88)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_tau.pdf"))


def fig_finetune():
    runs = [("Base", "base_test500.jsonl"), ("Filtered\n$\\tau$=0.9", "filtered_tau0.9_test500.jsonl"),
            ("No filter\n$\\tau$=0", "tau0_test500.jsonl"), ("Oracle\n(gold labels)", "oracle_tau0.9_test500.jsonl")]
    accs = [100 * sum(r["ok"] for r in load(f)) / 500 for _, f in runs]
    colors = [GRAY, BLUE, BLUE, AQUA]
    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    bars = ax.bar(range(len(runs)), accs, width=0.6, color=colors, edgecolor="white", linewidth=2)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.6, f"{a:.1f}", ha="center", va="bottom",
                fontsize=7, color=INK)
    ax.axhline(accs[0], color=MUTED, lw=1, ls="--", zorder=0)
    ax.set_xticks(range(len(runs)), [n for n, _ in runs])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(50, 82)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_finetune.pdf"))


def fig_confidence():
    base = {r["idx"]: r for r in load("base_test500.jsonl")}
    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    for name, color, ls, label in [("base_test500.jsonl", GRAY, "-", "Base"),
                                   ("filtered_tau0.9_test500.jsonl", BLUE, "-", "Filtered $\\tau$=0.9"),
                                   ("tau0_test500.jsonl", BLUE, "--", "No filter $\\tau$=0"),
                                   ("oracle_tau0.9_test500.jsonl", AQUA, "-", "Oracle")]:
        wrong = sorted(r["conf"] for r in load(name) if not r["ok"])
        n = len(wrong)
        ys = [100 * (i + 1) / n for i in range(n)]
        ax.step(wrong, ys, where="post", color=color, lw=2, ls=ls, label=f"{label} (n={n})")
    ax.set_xlabel("Confidence $c(x)$ on wrong answers")
    ax.set_ylabel("Cumulative % of errors")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_confidence.pdf"))
    assert len(base) == 500


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_tau()
    fig_finetune()
    fig_confidence()
    print("wrote", sorted(os.listdir(OUT)))
