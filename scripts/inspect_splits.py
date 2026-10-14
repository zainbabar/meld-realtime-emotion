"""Sanity-check the official MELD splits: row counts, label counts, examples, and clip coverage.

Usage:
  python -I scripts/inspect_splits.py --csv-dir DIR --mp4-dir train=DIR --mp4-dir dev=DIR --mp4-dir test=DIR
"""
import argparse
import re
from pathlib import Path

import pandas as pd

SPLITS = ("train", "dev", "test")
# Expected sizes from the MELD paper (Poria et al., 2019): utterances, dialogues.
EXPECTED = {"train": (9989, 1039), "dev": (1109, 114), "test": (2610, 280)}
EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
CLIP_RE = re.compile(r"^dia(\d+)_utt(\d+)\.mp4$")
TINY_BYTES = 1024


def clip_name(row):
    return f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--mp4-dir", action="append", default=[], help="split=DIR, once per split")
    args = ap.parse_args()
    mp4_dirs = {k: Path(v) for k, v in (s.split("=", 1) for s in args.mp4_dir)}

    dfs = {s: pd.read_csv(args.csv_dir / f"{s}_sent_emo.csv") for s in SPLITS}

    print("== rows and dialogues per split (expected from the MELD paper) ==")
    for s, df in dfs.items():
        n_utt, n_dia = len(df), df.Dialogue_ID.nunique()
        exp_utt, exp_dia = EXPECTED[s]
        flag = "" if (n_utt, n_dia) == (exp_utt, exp_dia) else "   <-- MISMATCH"
        print(f"  {s:5s}  rows {n_utt:5d} (exp {exp_utt})   dialogues {n_dia:4d} (exp {exp_dia}){flag}")
        dupes = df.duplicated(["Dialogue_ID", "Utterance_ID"]).sum()
        if dupes:
            print(f"         {dupes} duplicated (Dialogue_ID, Utterance_ID) pairs")
    print(f"  columns: {list(dfs['train'].columns)}")

    print("\n== emotion label counts per split ==")
    counts = pd.DataFrame({s: df.Emotion.value_counts() for s, df in dfs.items()}).reindex(EMOTIONS).fillna(0).astype(int)
    unexpected = set().union(*(set(df.Emotion) for df in dfs.values())) - set(EMOTIONS)
    pct = counts / counts.sum() * 100
    table = counts.astype(str) + pct.map(lambda p: f" ({p:4.1f}%)")
    table.loc["total"] = counts.sum().astype(str)
    print(table.to_string())
    if unexpected:
        print(f"  unexpected labels: {sorted(unexpected)}")

    print("\n== 10 example rows (train, first rows = first dialogues, in order) ==")
    cols = ["Dialogue_ID", "Utterance_ID", "Speaker", "Emotion", "Sentiment", "Utterance", "StartTime", "EndTime"]
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(dfs["train"][cols].head(10).to_string(index=False))

    print("\n== non-ASCII characters in utterances (possible encoding artifacts) ==")
    for s, df in dfs.items():
        odd = df.Utterance.str.contains(r"[^\x00-\x7f]", regex=True, na=False)
        chars = sorted(set("".join(df.Utterance[odd])) - set(map(chr, range(128))))
        print(f"  {s:5s}  {odd.sum():4d} utterances, chars: {''.join(chars)!r}")

    if not mp4_dirs:
        return
    print("\n== clip coverage (CSV rows vs mp4 files) ==")
    for s in SPLITS:
        if s not in mp4_dirs:
            continue
        d = mp4_dirs[s]
        files = {p.name: p for p in d.iterdir() if p.is_file()}
        wanted = {clip_name(r) + ".mp4" for r in dfs[s].itertuples()}
        missing = sorted(wanted - files.keys())
        tiny = sorted(n for n in wanted & files.keys() if files[n].stat().st_size < TINY_BYTES)
        extra = sorted(files.keys() - wanted)
        extra_clips = [n for n in extra if CLIP_RE.match(n)]
        print(f"  {s:5s}  dir {d}")
        print(f"         {len(files)} files, {len(wanted)} CSV rows, {len(missing)} missing, {len(tiny)} < {TINY_BYTES} B")
        for n in missing:
            print(f"         MISSING  {s}/{n}")
        for n in tiny:
            print(f"         TINY     {s}/{n} ({files[n].stat().st_size} B)")
        if extra:
            print(f"         {len(extra)} files not in the CSV ({len(extra_clips)} match diaX_uttY.mp4), e.g. {extra[:5]}")


if __name__ == "__main__":
    main()
