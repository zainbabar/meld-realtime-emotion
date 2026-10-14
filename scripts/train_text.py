# text-only baseline: fine-tune a pretrained text encoder to predict the 7 MELD emotions
# from the utterance alone (no audio, no prior lines). picks the epoch with the best
# dev macro F1. never touches test.
#
# usage:
#   uv run python -I scripts/train_text.py --csv-dir data/meld/csv --wav-dir data/wav/center \
#       --out-dir runs/text_base_s0 [--model FacebookAI/roberta-base] [--epochs 4] [--seed 0] [--no-class-weights]
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))  # let scripts import each other under -I
from meld_data import EMOTIONS, load_all


def class_weights(labels):
    # return a float tensor of shape (7,), one weight per class, so mistakes on
    # rare classes (disgust, fear) cost more in the loss. labels is a list of ints 0-6.
    # --no-class-weights skips this, so every class counts the same (plain cross-entropy)
    labels = torch.tensor(labels) # turn from plain python list to tensor
    n = labels.shape[0] # total # of training examples
    k = len(EMOTIONS) # number of classes
    counts = torch.bincount(labels, minlength=7) # count of hm times each label 0-6 appears, tensor shape (7, )
    weights = n / (k * counts)
    return weights



def make_loader(df, tokenizer, max_len, batch_size, shuffle):
    rows = list(zip(df.text, df.label))

    # tokenize one batch at a time, so each batch is only padded to its own longest line
    def collate(batch):
        texts = [text for text, label in batch]
        labels = [label for text, label in batch]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def predict(model, loader, device):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
            preds += logits.argmax(dim=-1).tolist()
            golds += labels.tolist()
    return np.array(preds), np.array(golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="FacebookAI/roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-class-weights", action="store_true", help="plain cross-entropy, for the unweighted baseline")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all(args.csv_dir, args.wav_dir)
    train, dev = data["train"], data["dev"]  # test stays untouched until the very end

    # pretrained encoder + a new, randomly initialized linear head with 7 outputs on top.
    # both get trained: the head learns the classes, the encoder adapts a little
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(EMOTIONS))
    model.to(device)

    train_loader = make_loader(train, tokenizer, args.max_len, args.batch_size, shuffle=True)
    dev_loader = make_loader(dev, tokenizer, args.max_len, args.batch_size, shuffle=False)

    weights = None
    if not args.no_class_weights:
        weights = class_weights(train.label.tolist())
    if weights is not None:
        weights = weights.to(device)
        print(f"class weights: {dict(zip(EMOTIONS, weights.tolist()))}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    # small learning rate so fine-tuning nudges the pretrained weights instead of wrecking them.
    # lr ramps up over the first 10% of steps, then falls linearly to 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = batch.to(device)
            # bf16 autocast: matmuls run in bfloat16 (much faster on the GB10), weights stay fp32
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        preds, golds = predict(model, dev_loader, device)
        macro = f1_score(golds, preds, average="macro")
        weighted = f1_score(golds, preds, average="weighted")
        secs = time.perf_counter() - t0
        print(f"epoch {epoch}  train loss {total_loss / len(train_loader):.4f}  "
              f"dev macro F1 {macro:.4f}  weighted F1 {weighted:.4f}  ({secs:.0f} s)")
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_loader),
                        "dev_macro_f1": macro, "dev_weighted_f1": weighted})

        # keep the epoch with the best dev macro F1, since macro is the number we care about
        if macro > best_f1:
            best_f1 = macro
            model.save_pretrained(args.out_dir / "best")
            tokenizer.save_pretrained(args.out_dir / "best")
            report = classification_report(golds, preds, target_names=EMOTIONS, digits=3, zero_division=0)

    print(f"\nbest dev macro F1 {best_f1:.4f}, per class:\n{report}")
    results = {"args": {k: str(v) for k, v in vars(args).items()}, "history": history,
               "best_dev_macro_f1": best_f1, "best_report": report}
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"saved {args.out_dir / 'best'} and {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
