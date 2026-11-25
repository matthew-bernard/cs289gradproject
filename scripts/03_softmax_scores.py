"""Corrected softmax over answer options: prefill " Answer:" and read P(letter) at the next position.

Writes one JSONL row per question with the prediction, the confidence (max renormalized
option probability), and the option mass on the full vocabulary. Gold labels are stored
for evaluation and diagnostics only. Resumes from an existing output file.
"""
import argparse, json, os, time
import torch
from common import (LETTERS, chat_text, load_model, load_processor, load_split,
                    option_token_ids, pick_device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--n", type=int, default=None, help="seeded subset size; omit for the whole split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory to evaluate")
    ap.add_argument("--no-image-splitting", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds, idx = load_split(args.split, args.n, args.seed)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["idx"] for line in f}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    device, dtype = pick_device()
    processor = load_processor(image_splitting=not args.no_image_splitting)
    model = load_model(device, dtype, args.adapter).eval()

    correct = total = 0
    start = time.time()
    todo = [i for i in idx if i not in done]
    for step, i in enumerate(todo, 1):
        ex = ds[i]
        inputs = processor(text=chat_text(processor, ex), images=ex["image"].convert("RGB"),
                           return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float()

        n = len(list(ex["choices"]))
        ids = option_token_ids(processor, n)
        mass = float(torch.softmax(logits, dim=-1)[ids].sum())
        probs = torch.softmax(logits[ids], dim=-1)
        k = int(torch.argmax(probs))
        pred, gold = LETTERS[k], LETTERS[ex["answer"]]
        correct += int(pred == gold)
        total += 1

        rec = {"idx": i, "pred": pred, "gold": gold, "ok": pred == gold,
               "conf": round(float(probs[k]), 6), "mass": round(mass, 6),
               "probs": {LETTERS[j]: round(float(p), 6) for j, p in enumerate(probs)}}
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if step % 50 == 0 or step == len(todo):
            rate = (time.time() - start) / step
            print(f"[{step}/{len(todo)}] acc={correct/total:.4f} {rate:.2f}s/q "
                  f"eta={rate*(len(todo)-step)/60:.1f}min", flush=True)

    rows = [json.loads(line) for line in open(args.out)]
    acc = sum(r["ok"] for r in rows) / len(rows)
    mass = sum(r["mass"] for r in rows) / len(rows)
    print(f"FINAL {args.split} n={len(rows)} acc={acc:.4f} mean_option_mass={mass:.4f}")


if __name__ == "__main__":
    main()
