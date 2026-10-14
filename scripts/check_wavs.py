# list every clip that failed or looks suspect after extract_audio.py
#
# failed:  missing mp4, ffmpeg error, empty wav (from the manifest)
# warning: ffmpeg prints anything at warning level while decoding the audio (a second, full decode)
# suspect: very short, near silent, or very long wavs, which usually mean bad csv timestamps
#
# writes <wav-dir>/suspect_clips.csv
#
# usage:
#   uv run python -I scripts/check_wavs.py --wav-dir data/wav/mix
import argparse
import os
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SHORT_S = 0.2     # shorter than about one syllable
QUIET_DB = -50.0  # rms in dBFS; typical clips sit around -31
LONG_S = 30.0
# ffmpeg says this for every 6-channel clip with no layout tag, it isn't a decode problem
IGNORE = "Guessed Channel Layout"


def decode_warnings(mp4):
    # decode the audio to nowhere and keep whatever ffmpeg complains about
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "warning", "-i", mp4, "-vn", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = [line for line in r.stderr.splitlines() if IGNORE not in line]
    return " | ".join(lines)


def rms_db(path):
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x = x.astype(np.float32) / 32768
    return 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", type=Path, required=True)
    args = ap.parse_args()
    m = pd.read_csv(args.wav_dir / "manifest.csv")

    print("== failed ==")
    failed = m[~m.status.isin(["ok", "ok_warnings"])]
    for r in failed.itertuples():
        print(f"  {r.status:12s} {r.split}/{r.name}  {r.message if isinstance(r.message, str) else ''}")

    print("\n== decode warnings (ffmpeg -v warning, full audio decode) ==")
    present = m[m.status != "missing"]
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        warnings = list(pool.map(decode_warnings, present.mp4))
    n_warned = 0
    for r, msg in zip(present.itertuples(), warnings):
        if msg:
            n_warned += 1
            print(f"  {r.split}/{r.name}  {msg[:200]}")
    print(f"  {n_warned} of {len(present)} clips")

    print("\n== suspect ==")
    ok = m[m.status.isin(["ok", "ok_warnings"])].copy()
    ok["rms_db"] = [rms_db(p) for p in ok.wav]
    reasons = []
    for r in ok.itertuples():
        why = []
        if r.duration_s < SHORT_S:
            why.append("short")
        if r.rms_db < QUIET_DB:
            why.append("quiet")
        if r.duration_s > LONG_S:
            why.append("long")
        reasons.append("+".join(why))
    ok["reason"] = reasons
    suspect = ok[ok.reason != ""]
    cols = ["split", "name", "Emotion", "duration_s", "rms_db", "reason"]
    suspect[cols].to_csv(args.wav_dir / "suspect_clips.csv", index=False)
    print(f"  {len(suspect)} of {len(ok)} clips: short < {SHORT_S} s, quiet < {QUIET_DB:.0f} dBFS, long > {LONG_S:.0f} s")
    print(pd.crosstab(suspect.reason, suspect.split, margins=True).to_string())
    print(f"  by emotion: {suspect.Emotion.value_counts().to_dict()}")
    print(f"  written to {args.wav_dir / 'suspect_clips.csv'}")


if __name__ == "__main__":
    main()
