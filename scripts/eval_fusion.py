# evaluate saved fusion checkpoints on one split, once. no training, no choices made here:
# every run in --runs is scored, and the mean over seeds is what gets reported.
# runs whose name contains "noaudio" are scored with the audio zeroed, the same way they were trained.
#
# usage:
#   uv run python -I scripts/eval_fusion.py --csv-dir data/meld/csv --wav-dir data/wav/center \
#       --emb-dir data/emb/wavlm_base_plus --split test \
#       --runs runs/fusion_s0 runs/fusion_s1 runs/fusion_s2 runs/fusion_noaudio_s0 runs/fusion_noaudio_s1 runs/fusion_noaudio_s2
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))  # let scripts import each other under -I
from audio_probe import standardize
from meld_data import EMOTIONS, load_all
from train_fusion import FusionModel, build_context, make_loader, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--model", default="FacebookAI/roberta-base")
    ap.add_argument("--out", type=Path, default=None, help="default runs/eval_<split>.json")
    ap.add_argument("--context", type=int, default=0, help="must match the --context the runs were trained with")
    ap.add_argument("--max-len", type=int, default=128, help="must match training too")
    args = ap.parse_args()
    device = torch.device("cuda")

    df = load_all(args.csv_dir, args.wav_dir)[args.split]
    # standardize with train stats, exactly like training
    audio_train = np.load(args.emb_dir / "train.npy").astype(np.float32)
    audio_split = np.load(args.emb_dir / f"{args.split}.npy").astype(np.float32)
    _, audio_split = standardize(audio_train, audio_split)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    contexts = None
    if args.context > 0:
        tokenizer.truncation_side = "left"
        contexts = build_context(df, args.context)

    results = {}
    for run in args.runs:
        audio = torch.zeros_like(audio_split) if "noaudio" in run.name else audio_split
        loader = make_loader(df, audio, tokenizer, args.max_len, 64, False, contexts)
        model = FusionModel(args.model, audio_split.shape[1], audio_split.shape[2])
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
        model.to(device)
        preds, golds = predict(model, loader, device)
        macro = f1_score(golds, preds, average="macro")
        weighted = f1_score(golds, preds, average="weighted")
        per_class = f1_score(golds, preds, average=None, labels=list(range(len(EMOTIONS))))
        report = classification_report(golds, preds, target_names=EMOTIONS, digits=3, zero_division=0)
        results[run.name] = {
            "macro_f1": macro,
            "weighted_f1": weighted,
            "per_class_f1": dict(zip(EMOTIONS, per_class.tolist())),
            "report": report,
        }
        print(f"{run.name:20s} {args.split} macro F1 {macro:.4f}  weighted F1 {weighted:.4f}")

    # mean over seeds for each setting: runs named <setting>_s<seed>, e.g. fusion_s0 and fusion_noaudio_s0
    print()
    settings = {}
    for name in results:
        settings.setdefault(name.rsplit("_s", 1)[0], []).append(name)
    for setting, names in settings.items():
        macro = [results[n]["macro_f1"] for n in names]
        weighted = [results[n]["weighted_f1"] for n in names]
        print(
            f"{setting:22s} mean over {len(names)} seeds: macro F1 {np.mean(macro):.4f} "
            f"(min {min(macro):.4f}, max {max(macro):.4f})  weighted F1 {np.mean(weighted):.4f}"
        )
        for emotion in EMOTIONS:
            vals = [results[n]["per_class_f1"][emotion] for n in names]
            print(f"    {emotion:9s} {np.mean(vals):.3f}")

    out = args.out or Path("runs") / f"eval_{args.split}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
