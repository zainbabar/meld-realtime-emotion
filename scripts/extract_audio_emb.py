# run every wav through a frozen speech encoder once and save one vector per clip and layer,
# so the audio and fusion models can train on cached vectors instead of raw audio
#
# writes, per split:
#   OUT_DIR/<split>.npy       float16, shape (rows, layers, hidden): mean over time, every layer
#   OUT_DIR/<split>_rows.csv  name, has_audio, n_frames in the same row order (the csv order)
# rows with no audio get all zeros, so they're still there and line up with the csv
#
# usage:
#   uv run python -I scripts/extract_audio_emb.py --csv-dir data/meld/csv --wav-dir data/wav/center \
#       --out-dir data/emb/wavlm_base_plus [--model microsoft/wavlm-base-plus] [--limit N]
import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModel

sys.path.insert(0, str(Path(__file__).parent))  # let scripts import each other under -I
from meld_data import SPLITS, load_all

SR = 16000
MAX_S = 30.0  # only 3 clips are longer (bad timestamps, up to 305 s); keep their first 30 s


def read_wav(path):
    # int16 samples -> float in [-1, 1], the scale the encoder was trained on
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return x.astype(np.float32) / 32768


def normalize_wav(x):
    # zero mean, unit variance per clip, same as the model's own feature extractor does
    return (x - x.mean()) / np.sqrt(x.var() + 1e-7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows per split (smoke test)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()  # frozen: turns off dropout, and we never call backward
    n_layers = model.config.num_hidden_layers + 1  # +1 for the input to the first transformer layer
    hidden = model.config.hidden_size
    # wavlm-base-plus takes the raw waveform, wavlm-large expects each clip at zero mean, unit variance
    normalize = AutoFeatureExtractor.from_pretrained(args.model).do_normalize
    print(f"{args.model}: {n_layers} layers x {hidden}, normalize waveform: {normalize}")

    data = load_all(args.csv_dir, args.wav_dir)
    for split in SPLITS:
        df = data[split]
        if args.limit:
            df = df.head(args.limit)
        embs = np.zeros((len(df), n_layers, hidden), dtype=np.float16)
        n_frames = np.zeros(len(df), dtype=np.int32)
        t0 = time.perf_counter()

        # one clip at a time: no padding, so every frame is real audio
        for i, r in enumerate(df.itertuples()):
            if not r.has_audio:
                continue
            x = read_wav(r.wav)[: int(MAX_S * SR)]
            if normalize:
                x = normalize_wav(x)
            x = torch.from_numpy(x).unsqueeze(0).to(device)  # shape (1, samples)
            with torch.no_grad():
                out = model(x, output_hidden_states=True)
            # hidden_states: one (1, frames, hidden) tensor per layer, about 50 frames per second
            layers = torch.stack(out.hidden_states).squeeze(1)  # (layers, frames, hidden)
            embs[i] = layers.mean(dim=1).cpu().numpy()  # average over time -> (layers, hidden)
            n_frames[i] = layers.shape[1]

        np.save(args.out_dir / f"{split}.npy", embs)
        rows = pd.DataFrame({"name": df.name.values, "has_audio": df.has_audio.values, "n_frames": n_frames})
        rows.to_csv(args.out_dir / f"{split}_rows.csv", index=False)
        secs = time.perf_counter() - t0
        print(
            f"{split:5s}  {len(df)} rows, {df.has_audio.sum()} with audio, saved {embs.shape} "
            f"in {secs:.0f} s, frames min {n_frames[n_frames > 0].min()} max {n_frames.max()}"
        )


if __name__ == "__main__":
    main()
