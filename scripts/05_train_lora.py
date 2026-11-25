"""LoRA fine-tuning of SmolVLM on (pseudo-)labeled ScienceQA training questions.

Each example is the evaluation prompt (ending in the assistant prefill " Answer:") followed by
the target " {label}<end_of_utterance>". The loss covers only the target tokens. LoRA adapters
go on the attention and MLP projections of the language model; the vision encoder is frozen.
"""
import argparse, json, os, random, time
import torch
from peft import LoraConfig, get_peft_model
from common import chat_text, load_model, load_processor, load_split, pick_device

END = "<end_of_utterance>"


def encode(processor, ex, label, device, dtype):
    prompt = chat_text(processor, ex)
    image = ex["image"].convert("RGB")
    full = processor(text=prompt + f" {label}{END}", images=image, return_tensors="pt")
    n_prompt = processor(text=prompt, images=image, return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100
    full = full.to(device)
    full["pixel_values"] = full["pixel_values"].to(dtype)
    return full, labels.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="JSONL of {idx, label} on the train split")
    ap.add_argument("--out", required=True, help="directory for the LoRA adapter")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8, help="effective batch size (batch of 1)")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-image-splitting", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    items = [json.loads(line) for line in open(args.train)]
    random.shuffle(items)
    if args.max_examples:
        items = items[:args.max_examples]

    ds, _ = load_split("train")
    device, dtype = pick_device()
    processor = load_processor(image_splitting=not args.no_image_splitting)
    model = load_model(device, dtype)
    config = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    total_updates = max(1, args.epochs * len(items) // args.grad_accum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda u: max(0.0, 1 - u / total_updates))

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "w")
    model.train()
    step, running, start = 0, 0.0, time.time()
    for epoch in range(args.epochs):
        for item in items:
            inputs, labels = encode(processor, ds[item["idx"]], item["label"], device, dtype)
            loss = model(**inputs, labels=labels).loss
            (loss / args.grad_accum).backward()
            running += loss.item()
            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update = step // args.grad_accum
                rec = {"update": update, "loss": running / args.grad_accum,
                       "lr": scheduler.get_last_lr()[0], "sec": round(time.time() - start, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if update % 10 == 0 or update == total_updates:
                    rate = (time.time() - start) / step
                    print(f"update {update}/{total_updates} loss={rec['loss']:.4f} "
                          f"{rate:.2f}s/ex eta={rate*(args.epochs*len(items)-step)/60:.1f}min",
                          flush=True)
                running = 0.0

    model.save_pretrained(args.out)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=2)
    print(f"saved adapter -> {args.out} ({(time.time()-start)/60:.1f} min)")


if __name__ == "__main__":
    main()
