# Real-time emotion for a character robot (MELD, text + audio)

A prototype that listens to speech as it arrives, tags each utterance with one of MELD's seven emotions, and has a local LLM reply in a tone that fits. Everything runs locally on one GPU. The full inference path is 3.16B parameters, and the emotion tag is ready about 0.2 s after the speaker stops.

The emotion model fine-tunes RoBERTa-large on the transcript plus the previous three lines of the conversation, and fuses it with frozen WavLM-large audio features. On the MELD test set it reaches 0.509 macro F1 (0.658 weighted). Conversation context was the biggest single gain. The voice reliably helps anger and sadness and hurts fear and disgust, and with the large encoders that trade nets out slightly negative on macro F1.

| Model (mean of 3 seeds) | Test macro F1 | Test weighted F1 | Dev macro F1 |
| --- | ---: | ---: | ---: |
| Base: RoBERTa-base + WavLM-base+, fused | 0.490 | 0.644 | 0.477 |
| Base, same model with the audio zeroed | 0.467 | 0.623 | 0.472 |
| Large: RoBERTa-large + WavLM-large, fused | 0.498 | 0.652 | 0.520 |
| Large, same model with the audio zeroed | 0.509 | 0.644 | 0.500 |
| **Large + 3 lines of context, fused (final)** | **0.509** | **0.658** | **0.551** |
| Large + context, same model with the audio zeroed | 0.514 | 0.653 | 0.546 |

I picked every setting on dev and ran the test set once per model. The final model is large + context, fused, because it had the best dev macro F1. The zeroed-audio rows use the exact same model and code with the audio input set to zero, so the only thing that changes is whether the model can hear the voice.

## How it works

```
wav stream, 100 ms chunks
  -> endpointer (loudness, 600 ms of silence ends an utterance)
  -> Whisper-medium.en              -> transcript ─┐
  -> WavLM-large, all 25 layers     -> audio vector ┼─> fusion model -> emotion + confidence
     last 3 transcripts             -> context ─────┘
  transcript + emotion -> Qwen3-1.7B -> one-sentence reply, streamed
```

1. **Audio in.** A wav file plays back at real speed in 100 ms chunks, standing in for a microphone. The loop doesn't know where sentences end.
2. **Endpointing.** Each chunk is speech if it's louder than -40 dBFS. After speech starts, 600 ms of silence in a row ends the utterance. Anything under 300 ms is dropped (coughs, clicks) and anything over 15 s is cut.
3. **Emotion.** Whisper transcribes the utterance. WavLM-large turns the audio into one vector per layer (mean over time), standardized with training-set statistics. RoBERTa-large reads the last three transcripts and the current one as a text pair, `<s> context </s></s> line </s>`, and the model takes its `<s>` vector. If the pair is longer than 128 tokens, only the context gets cut, oldest words first. The fusion head mixes the 25 WavLM layers with learned softmax weights, puts text and audio through their own LayerNorm, concatenates them, and runs a small MLP (2048 -> 256 -> 7).
4. **Reply.** Qwen3-1.7B gets the transcript as the user turn and the detected emotion in its system prompt, and writes one sentence that fits the mood without naming it.

"Real-time" here means conversational turn-taking: the robot waits for you to finish, then answers. The 600 ms silence window comes first, then the tag lands about 0.2 s later and the first reply word about 0.3 s later.

### One utterance through the system

From MELD dev dialogue 90, streamed with the final config:

```json
{"start": 20.1, "end": 24.5,
 "text": "And the fact that you dumped him because he hit on Phoebe",
 "emotion": "anger", "confidence": 0.331,
 "reply": "I'm sorry about that, but I can't control what others do.",
 "ms_to_tag": 198, "ms_to_first_word": 280, "ms_to_full_reply": 609,
 "true_emotion": "anger", "true_text": "Whoa, whoa. And the fact that you dumped him because he hit on Phoebe?"}
```

On that 7-line dialogue the final config tags 2 of 7 correctly, and the large model without context got 4. With context it calls three neutral questions ("Rachel, how did this happen?", "Where?") surprise. Seven lines is too few to read much into, but it's a real miss and I'm reporting it.

## What I found

- Context helped more than anything else I tried. Adding the previous three lines raised dev macro F1 from 0.520 to 0.551 for the fused model. On test the gain shrank to +1.1 (fused) and +0.5 (audio zeroed). Choosing the best of several setups on dev inflates its dev score, which is why test only ran once.
- The voice helps some emotions and hurts others, consistently. Across every model and both splits, audio raises F1 on anger and sadness and lowers it on fear and disgust. Test per-class F1, final model (fused vs zeroed audio): anger 0.522 vs 0.518, sadness 0.455 vs 0.434, fear 0.281 vs 0.328, disgust 0.300 vs 0.326. Macro F1 weighs all seven classes equally, and fear and disgust are the two rarest (50 and 68 test lines), so they decide the sign. With base models the net was +2.3 macro F1 on test; with large it's -1.0, and with context -0.5. Weighted F1 goes up with audio in every case.
- With context, the audio adds less on dev too: +0.5 macro F1 (+1.5, +0.1, -0.1 by seed), down from +2.0 without context. Some of what the voice told the model, the previous lines now tell it.
- Dev and test disagree on audio for the large models. Without context, audio added +2.0 macro F1 on dev on all three seeds and cost 1.0 on test on all three. Dev has only 22 disgust and 40 fear lines, so its macro F1 swings a lot.
- Training longer didn't help. Six epochs instead of four gave 0.514 without context (vs 0.520) and 0.550 with it (vs 0.551).
- Square-root class weights were the only weighting that beat none on every seed. Text-only RoBERTa-base on dev: no weights 0.448, inverse-frequency 0.451, square-root 0.463 (+1.2 to +2.0 on each seed). Full inverse weights traded too much neutral recall for disgust and fear recall.
- Emotion sits in WavLM's upper-middle layers. A linear probe on each frozen layer peaks at layer 9 of 13 for WavLM-base+ (0.334 dev macro F1) and layer 18 of 25 for WavLM-large (0.385). Audio alone is far below text (0.463) but well above chance (about 0.12).
- The zeroed-audio fusion model already beats my first text-only script (0.472 vs 0.463 dev), because the new head and its separate learning rate help on their own. Measured against that script instead of the matched control, the audio gain would have looked more than twice as big.

## Parameters and latency

All learned parameters on the inference path, counted from the loaded models:

| Component | Role | Params |
| --- | --- | ---: |
| RoBERTa-large + fusion head | emotion (text side + head) | 355.9M |
| WavLM-large | emotion (audio features) | 315.5M |
| Whisper-medium.en | speech to text | 763.9M |
| Qwen3-1.7B | reply | 1,720.6M |
| **Total** | | **3.156B** (limit 6B) |

Latency on dev dialogue 90 (7 utterances), measured from the end-of-utterance decision:

| Config | Params | Tag (median / max) | First reply word | Full reply | Peak GPU memory |
| --- | ---: | --- | --- | --- | ---: |
| Base: RoBERTa-base, WavLM-base+, Whisper-small, Qwen3-1.7B | 2.18B | 90 / 153 ms | 169 / 237 ms | 521 / 978 ms | 4.8 GiB |
| Large encoders, Whisper-medium, Qwen3-1.7B, no context | 3.16B | 196 / 303 ms | 295 / 387 ms | 688 / 891 ms | 7.8 GiB |
| **Final: same, with 3 lines of context** | **3.16B** | **197 / 303 ms** | **280 / 385 ms** | **645 / 1,004 ms** | **7.8 GiB** |
| Large, no context, Qwen3-4B | 5.46B | 196 / 307 ms | 320 / 418 ms | 1,028 / 1,321 ms | 12.1 GiB |

Context costs no measurable latency, since the longer RoBERTa input is still at most 128 tokens. I tried filling the budget with Qwen3-4B. Replies read a bit better but take 50% longer, and the reply model doesn't touch the emotion scores, so I kept the 1.7B. `--llm Qwen/Qwen3-4B` switches it back.

## Hardware and resources

ASUS Ascent GX10: NVIDIA GB10 (sm_121), 20 Arm cores, 121 GiB of memory shared by CPU and GPU, CUDA 13.0, PyTorch 2.14.1+cu130 (aarch64).

| Step | Time |
| --- | --- |
| Audio extraction to wav (ffmpeg, 13,708 clips, 20 cores) | about 80 s per downmix |
| WavLM vectors, all clips (one clip at a time) | 4.3 min base+, 6.7 min large |
| Linear probes, all layers + weighted sum | 25 s (base+) |
| Fusion training, 4 epochs | about 2 min base, 5 min large, 7 min large + context |
| Live loop, loading all models | 27 s (final config) |

Inference peak memory is in the latency table. I didn't record peak memory during training.

## Data

MELD's official train / dev / test splits: 9,989 / 1,109 / 2,610 utterances from *Friends*, labelled neutral, joy, surprise, anger, sadness, disgust or fear. Neutral is 42-48% of each split; disgust and fear are 2-4%.

- **Audio:** 16 kHz mono wav. 88% of clips are 5.1 surround; I keep only the center channel, which held a median 81% of the energy and was the loudest channel in all 50 clips I measured. That drops the rear channels (a median 27 dB quieter) and the front left and right, which partly copy the center.
- **Text:** about 3,800 utterances contain Windows-1252 punctuation that was decoded as Latin-1 (`company\x92s`). I re-decode those bytes and fold curly quotes and dashes to ASCII.
- **Context:** the previous three lines of the same dialogue, in CSV order, joined with spaces. No speaker names, since the live loop can't tell speakers apart. The first line of a dialogue gets empty context.
- **Bad clips:** `dev/dia110_utt7` has no video file and `train/dia125_utt3` is corrupt; both stay in with an all-zero audio vector. 204 clips have broken timestamps (some 0.06 s long, one 305 s), and I kept them.
- **Rows vs files:** 137 test files and 4 dev files have no CSV row and are ignored.

## Quick start

You need Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg, and a CUDA GPU. The MELD download is 10.9 GB.

```sh
uv sync --locked

# download and unpack MELD
bash scripts/download_meld.sh
tar -xzf data/raw/MELD.Raw.tar.gz -C data/meld
mkdir -p data/meld/train data/meld/dev data/meld/test data/meld/csv
tar -xzf data/meld/MELD.Raw/train.tar.gz -C data/meld/train
tar -xzf data/meld/MELD.Raw/dev.tar.gz -C data/meld/dev
tar -xzf data/meld/MELD.Raw/test.tar.gz -C data/meld/test --exclude='._*'
cp data/meld/MELD.Raw/dev_sent_emo.csv data/meld/MELD.Raw/test_sent_emo.csv data/meld/train/train_sent_emo.csv data/meld/csv/

# center-channel wavs, then WavLM-large vectors
uv run python -I scripts/extract_audio.py --csv-dir data/meld/csv \
    --mp4-dir train=data/meld/train/train_splits --mp4-dir dev=data/meld/dev/dev_splits_complete \
    --mp4-dir test=data/meld/test/output_repeated_splits_test --out-dir data/wav/center --downmix center
uv run python -I scripts/extract_audio_emb.py --csv-dir data/meld/csv --wav-dir data/wav/center \
    --model microsoft/wavlm-large --out-dir data/emb/wavlm_large

# train the final fusion model (one seed), then run the live demo on a dev dialogue
uv run python -I scripts/train_fusion.py --csv-dir data/meld/csv --wav-dir data/wav/center \
    --emb-dir data/emb/wavlm_large --model FacebookAI/roberta-large --lr 1e-5 --context 3 --seed 0 --out-dir runs/ctx_fusion_s0
uv run python -I scripts/live_loop.py --csv-dir data/meld/csv --wav-dir data/wav/center \
    --emb-dir data/emb/wavlm_large --ckpt runs/ctx_fusion_s0/best.pt --context 3 --dev-dialogue 90
```

`--wav your_16k_mono.wav` streams your own recording instead of a dev dialogue. The models download from Hugging Face on first use (about 7.5 GB for the final config). I run every script with `python -I` because `data/` holds downloaded files.

## Reproducing the tables

| Result | Command |
| --- | --- |
| Main table, context rows | `train_fusion.py ... --model FacebookAI/roberta-large --lr 1e-5 --context 3 --seed {0,1,2} [--no-audio]`, then `eval_fusion.py --split test --model FacebookAI/roberta-large --emb-dir data/emb/wavlm_large --context 3 --runs <the six runs>` |
| Main table, large rows | the same without `--context 3` |
| Main table, base rows | `extract_audio_emb.py --out-dir data/emb/wavlm_base_plus` (default model), `train_fusion.py --emb-dir data/emb/wavlm_base_plus --seed {0,1,2} [--no-audio]`, then `eval_fusion.py --split test --emb-dir data/emb/wavlm_base_plus --runs ...` |
| More epochs | `train_fusion.py ... --epochs 6 [--context 3]` |
| Class weights | `train_text.py --class-weights {none,inverse,sqrt} --seed {0,1,2}` |
| Layer probes | `audio_probe.py --emb-dir data/emb/wavlm_{base_plus,large} --epochs 500` |
| Center channel measurement | `channel_levels.py` (needs both `--downmix mix` and `--downmix center` wavs) |
| Latency and parameters | `live_loop.py ... --dev-dialogue 90 [--context 3]` |

`eval_fusion.py` prints the per-seed scores, the mean over seeds and per-class F1. Each training script picks its epoch by dev macro F1 and never reads test.

## Files

| Script | What it does |
| --- | --- |
| `download_meld.sh` | Downloads MELD with full TLS verification (the server leaves out its intermediate certificate) |
| `inspect_splits.py` | Checks row counts, labels, encoding artifacts and clip coverage against the paper |
| `extract_audio.py` | mp4 -> 16 kHz mono wav with ffmpeg, `--downmix mix` or `center` |
| `channel_levels.py` | Measures how loud each 5.1 channel is |
| `check_wavs.py` | Lists failed, warning and suspect clips |
| `meld_data.py` | Loads the CSVs, fixes the text, maps labels, attaches wav paths |
| `train_text.py` | Text-only RoBERTa baseline with class weights |
| `extract_audio_emb.py` | Frozen WavLM vectors for every clip and layer |
| `audio_probe.py` | Linear probes per WavLM layer, plus a learned layer mix |
| `train_fusion.py` | The fusion model, with `--context` for previous lines and `--no-audio` for the control |
| `eval_fusion.py` | Scores saved fusion models on dev or test |
| `live_loop.py` | The real-time loop: endpointer, Whisper, fusion model, Qwen3 reply |

## Pretrained and generated parts

- **Fine-tuned here:** RoBERTa-large (`FacebookAI/roberta-large`), trained end to end with the fusion head.
- **Trained from scratch here:** the fusion head (layer mix, two LayerNorms, MLP), about 0.5M parameters.
- **Frozen, used as-is:** WavLM-large (`microsoft/wavlm-large`), Whisper-medium.en (`openai/whisper-medium.en`), Qwen3-1.7B (`Qwen/Qwen3-1.7B`).
- **Generated:** the robot's replies come from Qwen3-1.7B with a two-line prompt. I didn't evaluate them beyond reading the demo output.
- None of these models were trained on MELD as far as their model cards say. I avoided `emotion2vec` and `j-hartmann/emotion-english-distilroberta-base` because their training data includes MELD.
- **AI tools:** I used an AI coding assistant for the data plumbing (download, ffmpeg extraction, loaders, model loading in the live loop) and for drafting this README. The modelling decisions, the class weights, probes, fusion head, endpointer and experiment design are mine.

## Limitations

- Audio doesn't improve macro F1 for the final model on test. It helps anger and sadness, and weighted F1, but costs fear and disgust.
- Loudness endpointing counts laughter and music as speech. The center channel reduces this but doesn't remove it. A small VAD model would be the fix.
- The demo streams MELD clips joined by digital silence, which is easier to segment than a real room. I didn't test it with a live microphone.
- It runs in one thread. While an utterance is being processed, new chunks wait in a buffer and get processed when the loop catches up.
- The emotion model is trained on MELD's transcripts but runs on Whisper's, for both the current line and the context. Transcription errors ("Where? But his apartment" for "At his apartment") feed straight into the emotion model.
- The dev set is small: 22 disgust and 40 fear lines make dev macro F1 noisy, which is why I report three seeds and paired differences.
- WavLM's layer mix barely moves in the fusion model. For large, every weight stays between 0.038 and 0.044, where an even mix is 0.040, so in practice it's close to an average of all layers.
- The replies are unevaluated and sometimes off ("I'm not sure where you're referring to" for "Where?").

## Left out

- Vision and the three-way extension, and RL. I focused on the text + audio core.
- Speaker names in the context, and more than three previous lines.
- Fine-tuning WavLM; it stays frozen so its vectors can be cached and each run takes minutes.
- Provisional tags while someone is still speaking.
- Live microphone input.
