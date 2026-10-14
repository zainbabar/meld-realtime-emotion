"""Extract 16 kHz mono 16-bit PCM wav from every MELD clip listed in the official CSVs.

One output per CSV row: OUT_DIR/<split>/dia<D>_utt<U>.wav. Resumable: valid existing
wavs are kept. Writes OUT_DIR/manifest.csv with one row per CSV row and a status:
  ok           wav written and non-empty
  ok_warnings  wav written, but ffmpeg logged decode errors (possibly partly corrupt)
  missing      no mp4 for this row
  ffmpeg_error ffmpeg failed (corrupt file, no audio stream, ...)
  empty        ffmpeg succeeded but the wav has 0 samples

Usage:
  python -I scripts/extract_audio.py --csv-dir DIR --mp4-dir train=DIR --mp4-dir dev=DIR \
      --mp4-dir test=DIR --out-dir data/wav [--limit N] [--ffmpeg PATH]
"""
import argparse
import os
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

SR = 16000
SPLITS = ("train", "dev", "test")
LONG_CLIP_S = 30.0


def wav_info(path):
    """Duration in seconds, after checking the format is what we asked ffmpeg for."""
    with wave.open(str(path)) as w:
        fmt = (w.getframerate(), w.getnchannels(), w.getsampwidth())
        if fmt != (SR, 1, 2):
            raise ValueError(f"unexpected wav format {fmt}")
        return w.getnframes() / SR


def extract_one(ffmpeg, src, dst):
    """Return (status, duration_s, message)."""
    if not src.exists():
        return "missing", None, ""
    if dst.exists():
        try:
            dur = wav_info(dst)
            if dur > 0:
                return "ok", dur, "kept existing"
        except (wave.Error, ValueError, EOFError):
            pass
    tmp = dst.with_suffix(".tmp.wav")
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src),
           "-vn",                  # drop the video stream
           "-ac", "1",             # downmix to mono
           "-ar", str(SR),         # resample (low-pass filter + decimate) to 16 kHz
           "-c:a", "pcm_s16le",    # uncompressed 16-bit little-endian samples
           str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    msg = " | ".join(r.stderr.strip().splitlines()[-3:])
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return "ffmpeg_error", None, msg
    tmp.rename(dst)
    dur = wav_info(dst)
    if dur == 0:
        return "empty", 0.0, msg
    return ("ok_warnings" if msg else "ok"), dur, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--mp4-dir", action="append", required=True, help="split=DIR, once per split")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows per split (smoke test)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    mp4_dirs = {k: Path(v) for k, v in (s.split("=", 1) for s in args.mp4_dir)}

    jobs = []
    for split in SPLITS:
        df = pd.read_csv(args.csv_dir / f"{split}_sent_emo.csv")
        if args.limit:
            df = df.head(args.limit)
        (args.out_dir / split).mkdir(parents=True, exist_ok=True)
        for r in df.itertuples():
            name = f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}"
            jobs.append(dict(split=split, name=name, Dialogue_ID=r.Dialogue_ID, Utterance_ID=r.Utterance_ID,
                             Emotion=r.Emotion, mp4=mp4_dirs[split] / f"{name}.mp4",
                             wav=args.out_dir / split / f"{name}.wav"))

    # ffmpeg does the work in its own process, so threads are enough to keep all cores busy.
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda j: extract_one(args.ffmpeg, j["mp4"], j["wav"]), jobs))

    manifest = pd.DataFrame(jobs)
    manifest = manifest.join(pd.DataFrame(results, columns=["status", "duration_s", "message"], index=manifest.index))
    manifest_path = args.out_dir / ("manifest_smoke.csv" if args.limit else "manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print(pd.crosstab(manifest.split, manifest.status, margins=True).to_string())
    bad = manifest[manifest.status != "ok"]
    print(f"\n{len(bad)} rows not plainly ok:")
    for r in bad.itertuples():
        print(f"  {r.status:12s} {r.split}/{r.name}  {r.message}")
    d = manifest.duration_s.dropna()
    print(f"\nduration (s): min {d.min():.2f}  median {d.median():.2f}  p95 {d.quantile(.95):.2f}  "
          f"max {d.max():.2f}  total {d.sum() / 3600:.2f} h")
    long = manifest[manifest.duration_s > LONG_CLIP_S].sort_values("duration_s", ascending=False)
    print(f"{len(long)} clips longer than {LONG_CLIP_S:.0f} s, e.g.: "
          + ", ".join(f"{r.split}/{r.name} ({r.duration_s:.0f}s)" for r in long.head(5).itertuples()))
    print(f"\nmanifest: {manifest_path}")


if __name__ == "__main__":
    main()
