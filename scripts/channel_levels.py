# how loud is the center channel of the 5.1 clips compared with the other channels?
# backs the downmix choice in extract_audio.py (mix vs center)
#
# 1. sample: decode N random 6-channel clips at full channel count and compare per-channel rms
# 2. all clips: compare the level of every center wav against its mix wav, to catch 5.1 clips
#    where the center channel is near silent (center-only would throw the speech away)
#
# usage:
#   uv run python -I scripts/channel_levels.py --manifest data/wav/mix/manifest.csv \
#       --mix-dir data/wav/mix --center-dir data/wav/center [--n 50] [--seed 0]
import argparse
import subprocess
import wave
from pathlib import Path

import numpy as np
import pandas as pd

SR = 16000
# ffmpeg's channel order for 5.1, also what it guesses for 6-channel clips with no layout tag
NAMES = ["FL", "FR", "FC", "LFE", "BL", "BR"]
QUIET_DB = 20.0  # flag clips whose center wav is this much quieter than the mix wav


def db(x):
    return 20 * np.log10(x + 1e-10)


def rms(x):
    return np.sqrt(np.mean(x**2))


def decode_6ch(mp4):
    # all six channels as float32, shape (samples, 6), resampled to 16 kHz like the wavs
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(mp4),
        "-vn",
        "-ar",
        str(SR),
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "-",
    ]
    r = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, 6)


def layout_tag(mp4):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channel_layout",
        "-of",
        "csv=p=0",
        str(mp4),
    ]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def read_wav(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return x.astype(np.float32) / 32768


def sample_levels(manifest, n, seed):
    six = manifest[(manifest.channels == 6) & manifest.status.isin(["ok", "ok_warnings"])]
    picked = six.sample(n, random_state=seed)

    rows = []
    for r in picked.itertuples():
        x = decode_6ch(r.mp4)
        row = {"clip": f"{r.split}/{r.name}", "layout": layout_tag(r.mp4)}
        levels = [db(rms(x[:, c])) for c in range(6)]
        for c in range(6):
            # level of each channel relative to the center channel, in dB
            row[NAMES[c]] = levels[c] - levels[2]
        # share of the clip's total energy that sits in the center channel
        energy = (x.astype(np.float64) ** 2).sum(axis=0)
        row["fc_share"] = energy[2] / energy.sum()
        row["fc_db"] = levels[2]
        # do the front left/right channels carry a copy of the center channel?
        row["corr_fl"] = np.corrcoef(x[:, 2], x[:, 0])[0, 1]
        row["corr_fr"] = np.corrcoef(x[:, 2], x[:, 1])[0, 1]
        row["fc_loudest"] = int(np.argmax(levels)) == 2
        rows.append(row)
    return pd.DataFrame(rows)


def all_levels(manifest, mix_dir, center_dir):
    six = manifest[(manifest.channels == 6) & manifest.status.isin(["ok", "ok_warnings"])]
    rows = []
    for r in six.itertuples():
        mix = read_wav(mix_dir / r.split / f"{r.name}.wav")
        center = read_wav(center_dir / r.split / f"{r.name}.wav")
        rows.append({"clip": f"{r.split}/{r.name}", "mix_db": db(rms(mix)), "center_db": db(rms(center))})
    df = pd.DataFrame(rows)
    df["center_minus_mix"] = df.center_db - df.mix_db
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--mix-dir", type=Path, required=True)
    ap.add_argument("--center-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    manifest = pd.read_csv(args.manifest)

    print(f"== {args.n} random 6-channel clips (seed {args.seed}) ==")
    df = sample_levels(manifest, args.n, args.seed)
    print(f"layout tags: {df.layout.value_counts().to_dict()}")
    print("\nchannel level relative to FC, dB (median, then 10th-90th percentile):")
    for name in NAMES:
        q10, q50, q90 = df[name].quantile([0.1, 0.5, 0.9])
        print(f"  {name:3s}  {q50:6.1f}   [{q10:6.1f}, {q90:6.1f}]")
    print(f"\nFC is the loudest channel in {df.fc_loudest.sum()}/{len(df)} clips")
    print(f"FC share of total energy: median {df.fc_share.median():.2f}, min {df.fc_share.min():.2f}")
    print(f"FC level, dBFS: median {df.fc_db.median():.1f}, min {df.fc_db.min():.1f}")
    print(f"correlation FC vs FL: median {df.corr_fl.median():.2f}   FC vs FR: median {df.corr_fr.median():.2f}")

    # same numbers split by layout tag; if the untagged clips used a different channel
    # order, FC would stop being the loud channel in that group
    print("\nby layout tag (medians):")
    by_tag = df.groupby("layout")[NAMES + ["fc_share", "corr_fl"]].median().round(2)
    by_tag["n"] = df.layout.value_counts()
    print(by_tag.to_string())

    print("\n10 quietest-center clips in the sample:")
    cols = ["clip", "layout", "fc_db", "FL", "FR", "BL", "BR", "fc_share", "corr_fl"]
    print(df.sort_values("fc_share")[cols].head(10).round(2).to_string(index=False))

    print("\n== every 6-channel clip: center wav vs mix wav ==")
    lv = all_levels(manifest, args.mix_dir, args.center_dir)
    q10, q50, q90 = lv.center_minus_mix.quantile([0.1, 0.5, 0.9])
    print(
        f"{len(lv)} clips. center minus mix, dB: median {q50:.1f}, 10th-90th [{q10:.1f}, {q90:.1f}], "
        f"min {lv.center_minus_mix.min():.1f}, max {lv.center_minus_mix.max():.1f}"
    )
    quiet = lv[lv.center_minus_mix < -QUIET_DB].sort_values("center_minus_mix")
    print(f"{len(quiet)} clips with center more than {QUIET_DB:.0f} dB below mix:")
    for r in quiet.itertuples():
        print(f"  {r.clip}  center {r.center_db:.1f} dBFS  mix {r.mix_db:.1f} dBFS")


if __name__ == "__main__":
    main()
