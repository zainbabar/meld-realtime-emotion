# extract 16 kHz mono 16-bit wav from every MELD clip listed in the official csvs
#
# one output per csv row: OUT_DIR/<split>/dia<D>_utt<U>.wav. resumable: valid existing
# wavs are kept. writes OUT_DIR/manifest.csv with one row per csv row and a status:
#   ok           wav written and non-empty
#   ok_warnings  wav written, but ffmpeg logged decode errors (possibly partly corrupt)
#   missing      no mp4 for this row
#   ffmpeg_error ffmpeg failed (corrupt file, no audio stream, ...)
#   empty        ffmpeg succeeded but the wav has 0 samples
#
# --downmix picks how 5.1 clips become mono:
#   mix     ffmpeg's standard downmix of all channels (-ac 1, LFE dropped)
#   center  front-center channel only for 5.1 clips, -ac 1 for stereo clips
#
# usage:
#   uv run python -I scripts/extract_audio.py --csv-dir DIR --mp4-dir train=DIR --mp4-dir dev=DIR \
#       --mp4-dir test=DIR --out-dir data/wav/mix --downmix mix [--limit N]
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


def wav_duration(path):
    # check the format is what we asked ffmpeg for, then return the length in seconds
    with wave.open(str(path)) as w:
        fmt = (w.getframerate(), w.getnchannels(), w.getsampwidth())
        if fmt != (SR, 1, 2):
            raise ValueError(f"unexpected wav format {fmt}")
        return w.getnframes() / SR


def count_channels(ffprobe, src):
    # channels in the first audio stream, 0 if ffprobe can't read the file
    cmd = [ffprobe, "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=channels", "-of", "csv=p=0", str(src)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return 0
    return int(r.stdout.strip())


def downmix_args(downmix, channels):
    # 38% of the 6-channel clips have no layout tag; ffmpeg then guesses 5.1
    # (FL FR FC LFE BL BR), so FC is the third channel either way
    if downmix == "center" and channels == 6:
        return ["-af", "pan=mono|c0=FC"]
    return ["-ac", "1"]


def extract_one(ffmpeg, ffprobe, downmix, src, dst):
    # returns (status, channels, duration_s, message)
    if not src.exists():
        return "missing", None, None, ""
    channels = count_channels(ffprobe, src)
    if dst.exists():
        try:
            dur = wav_duration(dst)
            if dur > 0:
                return "ok", channels, dur, "kept existing"
        except (wave.Error, ValueError, EOFError):
            pass

    # write to a temp name first so a crash never leaves a half-written wav behind
    tmp = dst.with_suffix(".tmp.wav")
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src),
           "-vn",                              # drop the video stream
           *downmix_args(downmix, channels),   # down to one channel
           "-ar", str(SR),                     # resample (low-pass filter + decimate) to 16 kHz
           "-c:a", "pcm_s16le",                # uncompressed 16-bit little-endian samples
           str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    msg = " | ".join(r.stderr.strip().splitlines()[-3:])
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return "ffmpeg_error", channels, None, msg
    tmp.rename(dst)
    dur = wav_duration(dst)
    if dur == 0:
        return "empty", channels, 0.0, msg
    if msg:
        return "ok_warnings", channels, dur, msg
    return "ok", channels, dur, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--mp4-dir", action="append", required=True, help="split=DIR, once per split")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--downmix", choices=["mix", "center"], required=True)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows per split (smoke test)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    mp4_dirs = {}
    for s in args.mp4_dir:
        split, d = s.split("=", 1)
        mp4_dirs[split] = Path(d)

    # one job per csv row
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

    def run(job):
        return extract_one(args.ffmpeg, args.ffprobe, args.downmix, job["mp4"], job["wav"])

    # ffmpeg does the work in its own process, so threads are enough to keep all cores busy
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, jobs))

    manifest = pd.DataFrame(jobs)
    results = pd.DataFrame(results, columns=["status", "channels", "duration_s", "message"], index=manifest.index)
    manifest = manifest.join(results)
    manifest["downmix"] = args.downmix
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
    examples = [f"{r.split}/{r.name} ({r.duration_s:.0f}s)" for r in long.head(5).itertuples()]
    print(f"{len(long)} clips longer than {LONG_CLIP_S:.0f} s, e.g.: {', '.join(examples)}")
    print(f"\nmanifest: {manifest_path}")


if __name__ == "__main__":
    main()
