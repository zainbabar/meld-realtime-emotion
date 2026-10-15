# live loop: audio arrives in 100 ms chunks at real speed, an endpointer decides when an utterance
# has ended, then whisper (text) + wavlm (audio vectors) -> fused model -> emotion tag, and
# qwen3 writes a short reply streamed word by word. one json line per utterance, with timings.
#
# input is a wav file played back in real time, standing in for a microphone. --dev-dialogue N
# builds the stream from MELD dev dialogue N (clips joined with 1 s of silence) and prints the
# true label next to each prediction.
#
# defaults are the final config: roberta-large + wavlm-large fused model, whisper-medium.en, qwen3-1.7b.
# the base config: --text-model FacebookAI/roberta-base --audio-model microsoft/wavlm-base-plus
#   --asr-model openai/whisper-small.en --emb-dir data/emb/wavlm_base_plus --ckpt runs/fusion_s0/best.pt
#
# usage:
#   uv run python -I scripts/live_loop.py --csv-dir data/meld/csv --wav-dir data/wav/center \
#       --emb-dir data/emb/wavlm_large --ckpt runs/large_fusion_s0/best.pt --dev-dialogue 90
#   uv run python -I scripts/live_loop.py ... --wav some_16k_mono.wav
import argparse
import json
import sys
import time
import wave
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

sys.path.insert(0, str(Path(__file__).parent))  # let scripts import each other under -I
from extract_audio_emb import normalize_wav
from meld_data import EMOTIONS, load_all
from train_fusion import FusionModel

SR = 16000
CHUNK_MS = 100
CHUNK = SR * CHUNK_MS // 1000  # 1600 samples
GAP_S = 1.0  # silence between clips in a --dev-dialogue stream


def read_wav(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return x.astype(np.float32) / 32768


def chunk_db(chunk):
    # loudness of one chunk in dBFS (0 = full scale, typical speech in these clips is around -30)
    return 20 * np.log10(np.sqrt(np.mean(chunk**2)) + 1e-10)


class Endpointer:
    # decides when an utterance has ended.
    # push() gets one chunk at a time (float32, CHUNK samples). it returns the whole utterance
    # (all its chunks joined) on the chunk where the utterance ends, otherwise None.
    #   threshold_db: chunks louder than this count as speech
    #   hangover_ms:  this much silence in a row ends an utterance
    #   min_ms:       shorter utterances are dropped (coughs, clicks)
    #   max_ms:       longer utterances are cut and emitted anyway
    def __init__(self, threshold_db=-40.0, hangover_ms=600, min_ms=300, max_ms=15000):
        self.threshold_db = threshold_db
        self.hangover_chunks = hangover_ms // 100
        self.min_chunks = min_ms // 100
        self.max_chunks = max_ms // 100
        self.cur_speak = []
        self.cur_silence = []
        self.waiting = True
        self.speaking = False

    def push(self, chunk):
        db = chunk_db(chunk)
        # if loud enough to speak then add speak o/w add to silence
        if db > self.threshold_db:
            self.cur_silence = []
            self.speaking = True
            self.waiting = False
            self.cur_speak.append(chunk)
        elif self.speaking:  # only count quiet if speaking, waiting do nothing
            self.cur_silence.append(chunk)
            self.cur_speak.append(chunk)  # short pauses stay in the audio instead of being cut out

        # hit our limit just return even if still speaking
        if len(self.cur_speak) >= self.max_chunks:
            self.waiting = True
            self.speaking = False
            output = np.concat(self.cur_speak)
            self.cur_speak = []
            self.cur_silence = []
            return output
        # had enough silence to end
        if len(self.cur_silence) >= self.hangover_chunks:
            # too short or too long, sub the silent chunks
            if len(self.cur_speak) - len(self.cur_silence) < self.min_chunks:
                self.cur_speak = []
                self.cur_silence = []
                self.waiting = True
                self.speaking = False
                return None
            else:  # its ifne return
                self.waiting = True
                self.speaking = False
                output = np.concat(self.cur_speak)
                self.cur_speak = []
                self.cur_silence = []
                return output


def build_dev_stream(csv_dir, wav_dir, dialogue):
    # join one dev dialogue's clips with GAP_S of silence, keep where each clip sits for the truth labels
    dev = load_all(csv_dir, wav_dir)["dev"]
    rows = dev[(dev.Dialogue_ID == dialogue) & dev.has_audio]
    gap = np.zeros(int(GAP_S * SR), dtype=np.float32)
    parts, truth, t = [gap], [], GAP_S
    for r in rows.itertuples():
        x = read_wav(r.wav)
        truth.append({"start": t, "end": t + len(x) / SR, "emotion": r.Emotion, "text": r.text})
        parts += [x, gap]
        t += len(x) / SR + GAP_S
    return np.concatenate(parts), truth


def match_truth(truth, start, end):
    # the clip that overlaps this utterance the most
    best, best_overlap = None, 0.0
    for clip in truth:
        overlap = min(end, clip["end"]) - max(start, clip["start"])
        if overlap > best_overlap:
            best, best_overlap = clip, overlap
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--emb-dir", type=Path, required=True, help="for the train mean/std of the wavlm vectors")
    ap.add_argument("--ckpt", type=Path, required=True, help="fused model state_dict from train_fusion.py")
    ap.add_argument("--wav", type=Path, default=None, help="16 kHz mono wav to stream")
    ap.add_argument("--dev-dialogue", type=int, default=None, help="stream this MELD dev dialogue instead")
    ap.add_argument("--text-model", default="FacebookAI/roberta-large")
    ap.add_argument("--audio-model", default="microsoft/wavlm-large")
    ap.add_argument("--asr-model", default="openai/whisper-medium.en")
    ap.add_argument("--llm", default="Qwen/Qwen3-1.7B")
    ap.add_argument(
        "--context", type=int, default=0, help="previous utterances to give the text model, same as training"
    )
    ap.add_argument("--hangover-ms", type=int, default=600)
    ap.add_argument("--threshold-db", type=float, default=-40.0)
    ap.add_argument("--out", type=Path, default=None, help="also write the json lines here")
    args = ap.parse_args()
    device = torch.device("cuda")

    # load everything once, before the stream starts
    t_load = time.perf_counter()
    # same standardizing as training: train mean and std per layer and dimension
    train_emb = np.load(args.emb_dir / "train.npy").astype(np.float32)
    n_audio_layers, audio_dim = train_emb.shape[1], train_emb.shape[2]
    audio_mean = torch.from_numpy(train_emb.mean(axis=0)).to(device)
    audio_std = torch.from_numpy(train_emb.std(axis=0) + 1e-6).to(device)
    del train_emb

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    fused = FusionModel(args.text_model, n_audio_layers, audio_dim)
    fused.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    fused.to(device).eval()

    wavlm = AutoModel.from_pretrained(args.audio_model).to(device).eval()
    normalize = AutoFeatureExtractor.from_pretrained(args.audio_model).do_normalize  # true for wavlm-large

    asr_proc = WhisperProcessor.from_pretrained(args.asr_model)
    asr = WhisperForConditionalGeneration.from_pretrained(args.asr_model, dtype=torch.float16).to(device).eval()

    llm_tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device).eval()

    # learned parameters on the inference path, for the 6B budget
    counts = {
        f"{args.text_model} + head": sum(p.numel() for p in fused.parameters()),
        args.audio_model: sum(p.numel() for p in wavlm.parameters()),
        args.asr_model: sum(p.numel() for p in asr.parameters()),
        args.llm: sum(p.numel() for p in llm.parameters()),
    }
    for name, n in counts.items():
        print(f"  {name:36s} {n / 1e6:8.1f} M")
    print(f"  {'total':36s} {sum(counts.values()) / 1e9:8.3f} B   (limit 6 B)")
    print(f"models loaded in {time.perf_counter() - t_load:.1f} s\n")

    def transcribe(audio):
        feats = asr_proc(audio, sampling_rate=SR, return_tensors="pt").input_features
        ids = asr.generate(feats.to(device, torch.float16))
        return asr_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    history = []  # earlier transcripts in this stream, for --context

    def classify(audio, text):
        # same audio vector as extract_audio_emb.py: every layer, mean over time
        if normalize:
            audio = normalize_wav(audio)
        x = torch.from_numpy(audio).unsqueeze(0).to(device)
        hidden = torch.stack(wavlm(x, output_hidden_states=True).hidden_states).squeeze(1)  # (layers, frames, dim)
        vec = (hidden.mean(dim=1) - audio_mean) / audio_std  # (layers, dim)
        if args.context > 0:
            tokenizer.truncation_side = "left"
            ctx = " ".join(history[-args.context :])
            enc = tokenizer([ctx], [text], truncation="only_first", max_length=128, return_tensors="pt").to(device)
        else:
            enc = tokenizer([text], truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fused(enc.input_ids, enc.attention_mask, vec.unsqueeze(0))
        return torch.softmax(logits.float(), dim=-1)[0].cpu()

    def reply(text, emotion, conf):
        # the person's words go in as the user turn; the detected emotion steers the robot's tone
        messages = [
            {
                "role": "system",
                "content": "You are a warm, friendly character robot talking with a person. Reply to what they "
                f"say in one short sentence. From their words and tone of voice they seem to feel {emotion} "
                f"({conf:.0%} sure); let that shape how you respond, without naming the emotion.",
            },
            {"role": "user", "content": text},
        ]
        prompt = llm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = llm_tok(prompt, return_tensors="pt").to(device)
        # generate runs in a thread and pushes text into the streamer as tokens come out
        streamer = TextIteratorStreamer(llm_tok, skip_prompt=True, skip_special_tokens=True)
        gen = dict(**inputs, streamer=streamer, max_new_tokens=40, do_sample=False)
        thread = Thread(target=llm.generate, kwargs=gen)
        thread.start()
        return streamer, thread

    # warm up: the first call of each model pays one-time gpu setup, which shouldn't count as latency
    with torch.no_grad():
        noise = (np.random.default_rng(0).standard_normal(SR) * 0.05).astype(np.float32)
        classify(noise, transcribe(noise))
        streamer, thread = reply("hello", "neutral", 0.9)
        for _ in streamer:
            pass
        thread.join()

    if args.dev_dialogue is not None:
        stream, truth = build_dev_stream(args.csv_dir, args.wav_dir, args.dev_dialogue)
    else:
        stream, truth = read_wav(args.wav), []
    print(f"streaming {len(stream) / SR:.1f} s of audio in {CHUNK_MS} ms chunks\n")

    endpointer = Endpointer(threshold_db=args.threshold_db, hangover_ms=args.hangover_ms)
    out_file = open(args.out, "w") if args.out else None
    torch.cuda.reset_peak_memory_stats()
    results = []

    t0 = time.perf_counter()
    n_chunks = len(stream) // CHUNK
    for i in range(n_chunks):
        # a chunk only exists once its 100 ms have passed. if we fell behind while processing,
        # the waiting chunks come through right away, like a mic buffer catching up
        target = t0 + (i + 1) * CHUNK_MS / 1000
        time.sleep(max(0.0, target - time.perf_counter()))
        utterance = endpointer.push(stream[i * CHUNK : (i + 1) * CHUNK])
        if utterance is None:
            continue

        t_emit = time.perf_counter()
        end_s = (i + 1) * CHUNK_MS / 1000
        start_s = end_s - len(utterance) / SR
        with torch.no_grad():
            text = transcribe(utterance)
            probs = classify(utterance, text)
        history.append(text)
        k = int(probs.argmax())
        t_tag = time.perf_counter()
        print(f'[{start_s:6.1f}-{end_s:6.1f} s] "{text}"')
        print(f"  emotion {EMOTIONS[k]} ({probs[k]:.2f})   {(t_tag - t_emit) * 1000:.0f} ms after the pause")

        streamer, thread = reply(text, EMOTIONS[k], float(probs[k]))
        print("  robot: ", end="", flush=True)
        words, t_first = [], None
        for piece in streamer:
            if t_first is None and piece.strip():
                t_first = time.perf_counter()
            print(piece, end="", flush=True)
            words.append(piece)
        thread.join()
        t_done = time.perf_counter()
        print()

        result = {
            "start": round(start_s, 2),
            "end": round(end_s, 2),
            "text": text,
            "emotion": EMOTIONS[k],
            "confidence": round(float(probs[k]), 3),
            "probs": {e: round(float(p), 3) for e, p in zip(EMOTIONS, probs)},
            "reply": "".join(words).strip(),
            "ms_to_tag": round((t_tag - t_emit) * 1000),
            "ms_to_first_word": round(((t_first or t_done) - t_emit) * 1000),
            "ms_to_full_reply": round((t_done - t_emit) * 1000),
        }
        clip = match_truth(truth, start_s, end_s)
        if clip:
            result["true_emotion"] = clip["emotion"]
            result["true_text"] = clip["text"]
            print(f'  truth: {clip["emotion"]}   "{clip["text"]}"')
        print(f"  first word {result['ms_to_first_word']} ms, full reply {result['ms_to_full_reply']} ms\n")
        results.append(result)
        if out_file:
            out_file.write(json.dumps(result) + "\n")

    if out_file:
        out_file.close()
    print(f"{len(results)} utterances, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    if results:
        for key in ["ms_to_tag", "ms_to_first_word", "ms_to_full_reply"]:
            vals = [r[key] for r in results]
            print(f"  {key:17s} median {np.median(vals):6.0f}   max {max(vals):6.0f}")


if __name__ == "__main__":
    main()
