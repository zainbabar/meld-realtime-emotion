# load the official MELD csvs into one clean table per split
#
# - fixes the windows-1252 punctuation that got decoded as latin-1 (company\x92s -> company's)
# - maps the 7 emotion labels to ids
# - attaches the wav path for each row and flags rows with no usable audio
#
# needs a wav folder made by extract_audio.py (it reads <wav-dir>/manifest.csv)
#
# usage (prints a summary to check it works):
#   uv run python -I scripts/meld_data.py --csv-dir data/meld/csv --wav-dir data/wav/mix
import argparse
import re
from pathlib import Path

import pandas as pd

SPLITS = ("train", "dev", "test")
# most to least frequent in train
EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
LABEL_ID = {e: i for i, e in enumerate(EMOTIONS)}

# U+0080-U+009F are control characters nobody types. in MELD they're cp1252 punctuation
# bytes that were read as latin-1, so the byte value is the cp1252 code
C1_RE = re.compile("[\u0080-\u009f]")
# then fold curly quotes and dashes to ascii, since half the corpus already uses
# ascii ' and " and we want "company's" to tokenize the same way everywhere
TO_ASCII = {
    "‘": "'", "’": "'",   # single quotes
    "“": '"', "”": '"',   # double quotes
    "–": "-", "—": "--",  # en dash, em dash
    "…": "...",                # ellipsis
    " ": " ",                  # non-breaking space
}
AUDIO_OK = ("ok", "ok_warnings")


def fix_c1(match):
    return bytes([ord(match.group())]).decode("cp1252")


def clean_text(text):
    text = C1_RE.sub(fix_c1, text)
    for bad, good in TO_ASCII.items():
        text = text.replace(bad, good)
    # collapse the double spaces after sentences
    text = " ".join(text.split())
    return text


def load_split(csv_dir, wav_dir, split):
    df = pd.read_csv(csv_dir / f"{split}_sent_emo.csv")
    df["split"] = split
    df["name"] = "dia" + df.Dialogue_ID.astype(str) + "_utt" + df.Utterance_ID.astype(str)
    df["text"] = df.Utterance.map(clean_text)
    df["speaker"] = df.Speaker.map(clean_text)
    df["label"] = df.Emotion.map(LABEL_ID)

    # audio status per row comes from extract_audio.py's manifest
    manifest = pd.read_csv(wav_dir / "manifest.csv")
    manifest = manifest[manifest.split == split][["name", "status", "duration_s"]]
    manifest = manifest.rename(columns={"status": "audio_status"})
    df = df.merge(manifest, on="name", how="left")  # left merge keeps the csv row order
    df["wav"] = [str(wav_dir / split / f"{name}.wav") for name in df.name]
    df["has_audio"] = df.audio_status.isin(AUDIO_OK)
    return df


def load_all(csv_dir, wav_dir):
    data = {}
    for split in SPLITS:
        data[split] = load_split(csv_dir, wav_dir, split)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, required=True)
    args = ap.parse_args()
    data = load_all(args.csv_dir, args.wav_dir)

    print("== rows, audio, labels ==")
    for split, df in data.items():
        no_audio = df[~df.has_audio]
        names = [f"{r.name} ({r.audio_status})" for r in no_audio.itertuples()]
        print(f"  {split:5s}  {len(df):5d} rows, {df.has_audio.sum():5d} with audio, "
              f"{len(no_audio)} without: {names}")
        print(f"         label id counts: {df.label.value_counts().sort_index().to_dict()}")
        print(f"         unmapped labels: {df.label.isna().sum()}")

    print("\n== text cleaning ==")
    for split, df in data.items():
        changed = df.text != df.Utterance
        non_ascii = df.text.str.contains(r"[^\x00-\x7f]", regex=True)
        chars = set()
        for text in df.text[non_ascii]:
            chars |= {c for c in text if ord(c) > 127}
        print(f"  {split:5s}  {changed.sum():4d} utterances changed, "
              f"{df.text.str.contains(C1_RE).sum()} still have C1 chars, "
              f"non-ascii left: {''.join(sorted(chars))!r}")
    train = data["train"]
    examples = train[train.Utterance.str.contains(C1_RE)].head(4)
    for r in examples.itertuples():
        print(f"  before {r.Utterance!r}")
        print(f"  after  {r.text!r}")

    print("\n== wav paths ==")
    for split, df in data.items():
        missing_files = 0
        for path in df.wav[df.has_audio]:
            if not Path(path).exists():
                missing_files += 1
        print(f"  {split:5s}  {missing_files} rows with has_audio=True but no wav on disk")

    print("\n== one row ==")
    cols = ["split", "name", "speaker", "text", "Emotion", "label", "wav", "audio_status", "duration_s", "has_audio"]
    print(train[cols].iloc[3].to_string())


if __name__ == "__main__":
    main()
