import random
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
DATASET = "lmms-lab/ScienceQA-IMG"
LETTERS = "ABCDEFGH"
PREFILL = " Answer:"


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def load_split(split, n=None, seed=0):
    """Return the dataset split and the row indices to use (all rows, or a seeded subset like 01/02)."""
    ds = load_dataset(DATASET, split=split)
    if n is None or n >= len(ds):
        return ds, list(range(len(ds)))
    random.seed(seed)
    return ds, sorted(random.sample(range(len(ds)), n))


def load_processor(image_splitting=True):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.do_image_splitting = image_splitting
    return processor


def load_model(device, dtype, adapter=None):
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).to(device)
    return model


def build_prompt(ex):
    choices = list(ex["choices"])
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"Question: {ex['question']}\n{opts}\n"
        f"Answer with a single letter ({', '.join(LETTERS[:len(choices)])})."
    )


def chat_text(processor, ex):
    """Chat-formatted prompt ending right after the assistant prefill " Answer:"."""
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": build_prompt(ex)},
    ]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True) + PREFILL


def option_token_ids(processor, n_choices):
    ids = []
    for L in LETTERS[:n_choices]:
        toks = processor.tokenizer.encode(" " + L, add_special_tokens=False)
        if len(toks) != 1:
            raise RuntimeError(f"' {L}' tokenizes to {toks}, not a single token")
        ids.append(toks[0])
    return ids
