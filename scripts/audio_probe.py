# audio-only probe: how much emotion is in the frozen WavLM vectors, and in which layer?
# trains a tiny classifier on the cached vectors (one per clip) for each of the 13 layers,
# plus one learned weighted sum over all layers, and reports dev macro F1 for each.
# never touches test.
#
# usage:
#   uv run python -I scripts/audio_probe.py --csv-dir data/meld/csv --wav-dir data/wav/center \
#       --emb-dir data/emb/wavlm_base_plus [--epochs 30] [--lr 1e-3] [--seed 0]
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))  # let scripts import each other under -I
from meld_data import EMOTIONS, load_all
from train_text import class_weights


def standardize(train_x, other_x):
    # TODO(zain): per-dimension standardize with the TRAIN mean and std only, so dev never
    # leaks into the stats. train_x is (n_train, ..., 768), other_x is (n_other, ..., 768).
    # return (train_x_std, other_x_std), same shapes as the inputs
    pass


class LinearProbe(nn.Module):
    # TODO(zain): one linear layer, 768 -> 7. input (B, 768), output logits (B, 7)
    pass


class WeightedSumProbe(nn.Module):
    # TODO(zain): 13 trainable layer weights (start at zero), softmax them, take the weighted
    # sum over the layer axis, then one linear layer 768 -> 7.
    # input (B, 13, 768), output logits (B, 7)
    pass


def train_probe(model, x_train, y_train, x_dev, y_dev, weights, epochs, lr, device):
    # TODO(zain): train on train, evaluate dev macro F1 after every epoch, return the best one.
    # x_* are float tensors, y_* are long tensors of label ids, weights is the (7,) class
    # weight tensor for the loss. model.eval() + no_grad for the dev pass
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # labels come from the csvs; the .npy rows are in the same order (extract_audio_emb.py wrote them that way)
    data = load_all(args.csv_dir, args.wav_dir)
    y_train = torch.tensor(data["train"].label.values)
    y_dev = torch.tensor(data["dev"].label.values)

    # (rows, 13 layers, 768), float16 on disk -> float32 for training.
    # the 2 rows with no audio are all zeros; they stay in, so dev has the same 1109 rows as the text runs
    x_train = np.load(args.emb_dir / "train.npy").astype(np.float32)
    x_dev = np.load(args.emb_dir / "dev.npy").astype(np.float32)
    print(f"train {x_train.shape}, dev {x_dev.shape}")

    # same sqrt class weights as the text baseline, so the numbers are comparable
    weights = torch.sqrt(class_weights(data["train"].label.tolist())).to(device)

    # TODO(zain):
    # 1. standardize x_train and x_dev
    # 2. for each layer 0..12: slice that layer out, train a fresh LinearProbe, print
    #    f"layer {layer:2d}  dev macro F1 {f1:.4f}"
    # 3. train one WeightedSumProbe on all 13 layers, print its dev macro F1 and the 13
    #    learned layer weights (softmaxed), so we can see which layers it picked


if __name__ == "__main__":
    main()
