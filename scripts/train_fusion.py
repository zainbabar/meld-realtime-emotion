# text + audio fusion: fine-tune roberta with the frozen WavLM vectors joined in before the head.
# roberta's <s> vector (768) + a learned weighted sum of the 13 WavLM layers (768) -> head -> 7 emotions.
# same training setup as train_text.py (sqrt class weights, AdamW, warmup, clipping), so the
# numbers are comparable. picks the epoch with the best dev macro F1. never touches test.
#
# --no-audio zeroes the audio input: same model, same code, text only. that's the fair control
# for "does audio help", since only the audio changes
#
# usage:
#   uv run python -I scripts/train_fusion.py --csv-dir data/meld/csv --wav-dir data/wav/center \
#       --emb-dir data/emb/wavlm_base_plus --out-dir runs/fusion_s0 [--seed 0] [--no-audio] [--head-lr 1e-3]
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))  # let scripts import each other under -I
from audio_probe import standardize
from meld_data import EMOTIONS, load_all
from train_text import class_weights


class FusionHead(nn.Module):
    # inputs:  text_vec (B, text_dim)               roberta's <s> vector for each line
    #          audio    (B, n_audio_layers, audio_dim) standardized WavLM vectors, every layer
    # output:  logits   (B, 7)
    # sizes: base models are 768 text, 13 x 768 audio; large are 1024 text, 25 x 1024 audio
    # learned weighted sum over the audio layers (same as WeightedSumProbe) -> (B, audio_dim),
    # joined with the text side by side -> (B, text_dim + audio_dim), then a small MLP down to 7 scores
    def __init__(self, n_audio_layers=13, text_dim=768, audio_dim=768, n_outputs=7):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(n_audio_layers).reshape(1, n_audio_layers, 1))
        # one LayerNorm per half, so text and audio each get their own learned scale and shift.
        # modules set as attributes are registered automatically, no nn.Parameter needed
        self.audio_norm = nn.LayerNorm(audio_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.mlp = nn.Sequential(
            nn.Linear(text_dim + audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # zero 20% of the hidden values in training to prevent overfitting
            nn.Linear(256, n_outputs),
        )

    def forward(self, text_vec, audio):
        weights = torch.softmax(self.layer_weights, dim=1)
        weighted_sum = torch.sum((audio * weights), dim=1)  # weighted sum to encode each audio layers meaning

        # put both halves on the same scale, then join side by side -> (B, text_dim + audio_dim)
        combined = torch.concat((self.audio_norm(weighted_sum), self.text_norm(text_vec)), dim=1)
        return self.mlp(combined)


class FusionModel(nn.Module):
    # bare roberta (no built-in head) + fusion head. the audio sizes come from the cached vectors
    def __init__(self, model_name, n_audio_layers=13, audio_dim=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.head = FusionHead(n_audio_layers, self.encoder.config.hidden_size, audio_dim)

    def forward(self, input_ids, attention_mask, audio):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_vec = out.last_hidden_state[:, 0, :]  # (B, L, 768) -> the <s> row of each line, (B, 768)
        return self.head(text_vec, audio)


def build_context(df, n):
    # the previous n lines of the same dialogue, oldest first, joined with spaces. no speaker
    # names, since the live loop can't tell speakers apart. the csvs are in dialogue/utterance order
    contexts = []
    history = {}
    for r in df.itertuples():
        earlier = history.setdefault(r.Dialogue_ID, [])
        contexts.append(" ".join(earlier[-n:]))
        earlier.append(r.text)
    return contexts


def make_loader(df, audio, tokenizer, max_len, batch_size, shuffle, contexts=None):
    # each item is (text, label, row index); the index picks that line's audio from the big tensor
    # and that line's context, if any
    rows = list(zip(df.text, df.label, range(len(df))))

    def collate(batch):
        texts = [text for text, label, i in batch]
        labels = [label for text, label, i in batch]
        idx = [i for text, label, i in batch]
        if contexts is None:
            enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        else:
            # text pair: <s> context </s></s> line </s>. if too long, only the context gets cut,
            # from the left (oldest words first) because of truncation_side="left" set in main
            ctx = [contexts[i] for i in idx]
            enc = tokenizer(ctx, texts, padding=True, truncation="only_first", max_length=max_len, return_tensors="pt")
        enc["audio"] = audio[idx]  # (B, n_audio_layers, audio_dim)
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
                logits = model(**batch)
            preds += logits.argmax(dim=-1).tolist()
            golds += labels.tolist()
    return np.array(preds), np.array(golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="FacebookAI/roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5, help="roberta (pretrained) learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="fusion head (new weights) learning rate")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--context", type=int, default=0, help="previous lines of the dialogue to prepend (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-audio", action="store_true", help="zero the audio input (text-only control)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all(args.csv_dir, args.wav_dir)
    train, dev = data["train"], data["dev"]  # test stays untouched until the very end

    # cached WavLM vectors, standardized with train stats only (same as audio_probe.py)
    audio_train = np.load(args.emb_dir / "train.npy").astype(np.float32)
    audio_dev = np.load(args.emb_dir / "dev.npy").astype(np.float32)
    audio_train, audio_dev = standardize(audio_train, audio_dev)
    if args.no_audio:
        audio_train = torch.zeros_like(audio_train)
        audio_dev = torch.zeros_like(audio_dev)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = FusionModel(args.model, audio_train.shape[1], audio_train.shape[2]).to(device)

    train_ctx, dev_ctx = None, None
    if args.context > 0:
        tokenizer.truncation_side = "left"
        train_ctx, dev_ctx = build_context(train, args.context), build_context(dev, args.context)
    train_loader = make_loader(train, audio_train, tokenizer, args.max_len, args.batch_size, True, train_ctx)
    dev_loader = make_loader(dev, audio_dev, tokenizer, args.max_len, args.batch_size, False, dev_ctx)

    # sqrt class weights, the text baseline's default
    weights = torch.sqrt(class_weights(train.label.tolist())).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    # two learning rates: pretrained roberta only needs small nudges, the new head (and its 13
    # layer-mix weights) needs real steps. at 2e-5 for everything the mix never moved off 1/13
    param_groups = [
        {"params": model.encoder.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch)
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
        mix = torch.softmax(model.head.layer_weights, dim=1).flatten().detach().cpu()
        secs = time.perf_counter() - t0
        print(
            f"epoch {epoch}  train loss {total_loss / len(train_loader):.4f}  "
            f"dev macro F1 {macro:.4f}  weighted F1 {weighted:.4f}  ({secs:.0f} s)"
        )
        print("  layer mix: " + "  ".join(f"{i}:{w:.3f}" for i, w in enumerate(mix.tolist())))
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_loader),
                "dev_macro_f1": macro,
                "dev_weighted_f1": weighted,
                "layer_mix": mix.tolist(),
            }
        )

        # keep the epoch with the best dev macro F1
        if macro > best_f1:
            best_f1 = macro
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            report = classification_report(golds, preds, target_names=EMOTIONS, digits=3, zero_division=0)

    print(f"\nbest dev macro F1 {best_f1:.4f}, per class:\n{report}")
    results = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "history": history,
        "best_dev_macro_f1": best_f1,
        "best_report": report,
    }
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"saved {args.out_dir / 'best.pt'} and {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
