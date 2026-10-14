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
    # per-dimension standardize with the TRAIN mean and std only, so dev never
    # leaks into the stats. train_x is (n_train, ..., 768), other_x is (n_other, ..., 768).
    # return (train_x_std, other_x_std), same shapes as the inputs
    train_x_std = ((train_x - train_x.mean(axis=0)) / (train_x.std(axis=0) + 1e-6)) # add 1e-6 to avoid div by 0
    other_x_std = ((other_x - train_x.mean(axis=0)) / (train_x.std(axis=0) + 1e-6)) # std w train mean and std
    # convert to torch tensors 
    train_x_std = torch.from_numpy(train_x_std).float()
    other_x_std = torch.from_numpy(other_x_std).float()
    return (train_x_std, other_x_std)


class LinearProbe(nn.Module):
    # one linear layer, 768 -> 7. input (B, 768), output logits (B, 7)
    def __init__(self, n_inputs=768, n_outputs=7):
        super().__init__()
        self.layers = nn.Sequential( # just 1 layer for now, outputs 7 logits
            nn.Linear(in_features=n_inputs, out_features=n_outputs)
        )
    
    def forward(self, inputs):
        return self.layers(inputs)


class WeightedSumProbe(nn.Module):
    # 13 trainable layer weights (start at zero), softmax them, take the weighted
    # sum over the layer axis, then one linear layer 768 -> 7.
    # input (B, 13, 768), output logits (B, 7)
    def __init__(self, n_inputs=768, n_audio_layers=13, n_outputs=7):
        super().__init__()
        self.n_audio_layers = n_audio_layers
        # reshape this so its (1, 13, 1), so each layer has a weight 
        # wrapping in nn.Parameter() pytorch sees as part of the model so optimizer updates & gradients calc
        self.layer_weights = nn.Parameter(torch.zeros(n_audio_layers).reshape(1, n_audio_layers, 1))
        self.linear = nn.Linear(n_inputs, n_outputs)
    
    def forward(self, inputs):
        # inputs is shape (B, 13, 768), weights is shape (1, 13, 1)
        # mult by weights and take weighted sum on dim 1 so all layers
        # collapses into (B, 768) for linear 

        # positive layer mix that sums to 1; zeros start as an even average
        weights = torch.softmax(self.layer_weights, dim=1)
        weighted_sum = torch.sum((inputs * weights), dim=1)
        return self.linear(weighted_sum)
        

def train_probe(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor,
                x_dev: torch.Tensor, y_dev: torch.Tensor, weights: torch.Tensor,
                epochs: int, lr: float, device: torch.device) -> float:
    # train on train, evaluate dev macro F1 after every epoch, return the best one.
    # model:   fresh LinearProbe or WeightedSumProbe
    # x_train: float (9989, 768) for one layer, or (9989, 13, 768) for the weighted sum
    # y_train: long (9989,) label ids 0-6
    # x_dev:   float (1109, 768) or (1109, 13, 768), same layout as x_train
    # y_dev:   long (1109,)
    # weights: float (7,) sqrt class weights for the loss, already on device
    # returns: best dev macro F1 over the epochs

    # move everything to the gpu once, not every epoch
    model = model.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_dev = x_dev.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)  # same loss every time
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # fresh optimizer for this model's params

    best_f1 = -1.0
    for epoch in range(1, epochs + 1):
        # no dataloader: the whole train set is one batch, so one step per epoch
        model.train()
        logits = model(x_train)  # cross-entropy takes raw logits
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()  # compute gradients
        optimizer.step()  # update based on grad and lr

        # dev pass: eval mode, no gradients
        model.eval()
        with torch.no_grad():
            dev_logits = model(x_dev)
            dev_preds = torch.argmax(dev_logits, dim=1)  # take preds aka idx of highest logit
            dev_f1 = f1_score(y_dev, dev_preds.cpu(), average="macro")
        if dev_f1 > best_f1:
            best_f1 = dev_f1

    return best_f1




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

    # standardize x_train and x_dev
    # for each layer 0..12: slice that layer out, train a fresh LinearProbe, print
    # train one WeightedSumProbe on all 13 layers, print its dev macro F1 and the 13
    #learned layer weights (softmaxed), so we can see which layers it picked
    x_train_std, x_dev_std = standardize(x_train, x_dev)

    for layer in range(13):
        # slice out each layer from set
        train = x_train_std[:, layer, :]
        dev = x_dev_std[:, layer, :]
        probe = LinearProbe()
        f1 = train_probe(probe, train, y_train, dev, y_dev, weights, args.epochs, args.lr, device)
        print(f"layer {layer:2d} dev macro F1 {f1:.4f}")
    
    weighted_sum_probe = WeightedSumProbe()
    f1 = train_probe(weighted_sum_probe, x_train_std, y_train, x_dev_std, y_dev, weights, args.epochs, args.lr, device)
    print(f"weighted sum dev macro F1 {f1:.4f}")
    # softmax the raw (1, 13, 1) params the same way forward does, to get the final layer mix
    mix = torch.softmax(weighted_sum_probe.layer_weights, dim=1).flatten().detach().cpu()
    print("layer weights: " + "  ".join(f"{i}:{w:.3f}" for i, w in enumerate(mix.tolist())))


if __name__ == "__main__":
    main()
