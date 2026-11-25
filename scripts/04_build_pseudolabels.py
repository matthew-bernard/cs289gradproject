"""Build training sets from softmax scores on the train split, and report pseudo-label quality.

From one scores file (output of 03_softmax_scores.py on the train split) this writes three
training sets that share the same format, {"idx": ..., "label": ...}:
  filtered_tau{T}.jsonl  questions with conf >= tau, labeled with the model's own prediction
  tau0.jsonl             all questions, labeled with the model's own prediction (no filtering)
  oracle_tau{T}.jsonl    the same questions as filtered, labeled with the ground truth
Gold labels are read only to build the oracle set and to print the quality table.
"""
import argparse, json, os


def quality_table(rows, taus):
    print(f"{'tau':>6} {'kept':>6} {'coverage':>9} {'pseudo-label acc':>17}")
    for t in taus:
        kept = [r for r in rows if r["conf"] >= t]
        acc = sum(r["ok"] for r in kept) / len(kept) if kept else float("nan")
        print(f"{t:>6.2f} {len(kept):>6d} {len(kept)/len(rows):>9.3f} {acc:>17.4f}")


def write(path, items):
    with open(path, "w") as f:
        for idx, label in items:
            f.write(json.dumps({"idx": idx, "label": label}) + "\n")
    print(f"wrote {len(items):5d} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tau", type=float, default=None, help="threshold for the filtered/oracle sets")
    ap.add_argument("--outdir", default="data/pseudolabels")
    ap.add_argument("--table", type=float, nargs="*",
                    default=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.scores)]
    print(f"{args.scores}: n={len(rows)} acc={sum(r['ok'] for r in rows)/len(rows):.4f}")
    quality_table(rows, args.table)

    if args.tau is None:
        return
    os.makedirs(args.outdir, exist_ok=True)
    kept = [r for r in rows if r["conf"] >= args.tau]
    tag = f"tau{args.tau:g}"
    write(os.path.join(args.outdir, f"filtered_{tag}.jsonl"), [(r["idx"], r["pred"]) for r in kept])
    write(os.path.join(args.outdir, f"oracle_{tag}.jsonl"), [(r["idx"], r["gold"]) for r in kept])
    write(os.path.join(args.outdir, "tau0.jsonl"), [(r["idx"], r["pred"]) for r in rows])


if __name__ == "__main__":
    main()
