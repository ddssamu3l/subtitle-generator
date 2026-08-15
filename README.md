# Subtitle Generator

Burns speaker-aware subtitles into your videos, translated for **meaning** rather than word-for-word — and runs entirely on your own machine, with the wifi off.

Pick some videos, choose a subtitle language for each, and get back a new video with the subtitles rendered into the picture.

---

## What makes it different

**It translates intent, not words.** Cue-by-cue machine translation has no idea what the scene is about. This sends a rolling window of surrounding dialogue and speaker structure to a local LLM and asks it to carry across the *meaning* — idiom, register, and tone included.

> "I was starting to think you'd bailed on me."
> → 我快以为你**放鸽子**了。 *(the actual Chinese idiom for standing someone up — not a literal rendering)*
>
> "So, what's the damage?"
> → 那，一共**多少钱**？ *(correctly understood as asking for the bill)*

**Lines stack when people talk over each other.** A subtitle stays on screen long enough to read, which is usually longer than it took to say. If someone else starts speaking while the previous line is still up, the new line takes the bottom slot and the older one rises above it. If the *same* person keeps talking, their line is simply replaced. No "Speaker 1:" labels — just the layout doing the work.

```
   ┌────────────────────────────┐     ┌────────────────────────────┐
   │                            │     │   Traffic was a nightmare. │  ← earlier line rises
   │   You made it! I thought   │  →  │   You made it! I thought…  │
   │   you'd bailed on me.      │     │   Sorry I'm late.          │  ← new speaker takes bottom
   └────────────────────────────┘     └────────────────────────────┘
```

**It is genuinely offline.** Download the models once, then nothing ever leaves your machine — no audio, no transcript, no telemetry. Normal runs make zero network calls; that is enforced in code, not just promised.

---

## Requirements

| | |
|---|---|
| **Python 3.10+** | The installer handles this for you |
| **ffmpeg** | Reads your videos and renders the output |
| **Ollama** *(optional)* | Only needed for translation. Without it you still get subtitles in the spoken language. |
| **Disk** | ~3 GB for the transcription model, ~35 MB for speaker detection |

Works on macOS, Linux, and Windows. No GPU required. No PyTorch — the install is ~200 MB of wheels, not 2.5 GB.

---

## Install

```bash
git clone https://github.com/ddssamu3l/subtitle-generator.git
cd subtitle-generator
./install.sh          # Windows: .\install.ps1
```

That's it. The script installs [uv](https://docs.astral.sh/uv/) if you don't have it, creates an isolated environment with a Python that actually has a working GUI toolkit, installs the package, checks for ffmpeg, and downloads the models.

<details>
<summary><b>Manual install</b> (if you'd rather do it yourself)</summary>

```bash
uv venv --python 3.12          # a Python that bundles tkinter — see Troubleshooting
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e .
subgen setup                   # one-time model download (needs internet)
```

Without `uv`, any Python 3.10+ with working `tkinter` will do:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
subgen setup
```
</details>

### Enable translation (optional)

Translation runs through [Ollama](https://ollama.com/download) on your machine. We deliberately **don't** install it for you — it registers a background service and needs admin rights, which a subtitle tool has no business doing without being asked.

```bash
# 1. Install Ollama:  https://ollama.com/download   (or: brew install ollama)
# 2. Pull any chat model you like:
ollama pull qwen3:8b          # ~5 GB, strong at Chinese ↔ English
```

**Any chat-capable model works.** The tool lists whatever you have installed and lets you pick — no model name is hardcoded anywhere, so this keeps working as new models come out and today's are forgotten. Check what it can see:

```bash
subgen --list-models
```

---

## Use it

```bash
subgen
```

The app window opens. Click **Choose videos…** to open your file explorer, filtered to formats we can actually process. Pick one or many, set a subtitle language for each, then press **Generate subtitles**.

Output lands next to the original as `yourvideo.zh-Hans.mp4`. **Your original file is never modified.**

### Command line

```bash
subgen movie.mp4 --lang zh-Hans           # translate to Simplified Chinese
subgen *.mkv --lang en                    # batch, all to English
subgen clip.mp4                           # keep the spoken language
subgen clip.mp4 --lang ja --model qwen3:8b --scale 1.2
```

| Command | What it does |
|---|---|
| `subgen --status` | What's installed, what's missing, whether you can run offline |
| `subgen --list-models` | Ollama models available for translation |
| `subgen setup` | Download models for offline use (run once, online) |
| `subgen --purge` | Delete every cached model |

<details>
<summary>All options</summary>

| Flag | Meaning |
|---|---|
| `-l, --lang` | Target subtitle language (`zh-Hans`, `en`, `ja`, …). Omit to keep the spoken language. |
| `--spoken` | Override spoken-language detection |
| `-m, --model` | Ollama model to translate with |
| `--whisper-model` | Transcription model (`tiny`…`large-v3`, default `large-v3`) |
| `-o, --output-dir` | Where finished videos go |
| `--no-speakers` | Disable speaker detection, so lines never stack |
| `--no-sidecars` | Don't write `.ass`/`.srt` alongside the video |
| `--scale` | Subtitle size multiplier |
</details>

**Supported formats:** mp4, mov, mkv, webm, avi, m4v, mpg, mpeg, ts, m2ts, mts, wmv, flv, ogv, 3gp

---

## How it works

```
video ─→ ffmpeg ─→ 16kHz audio ─┬─→ Whisper ──────→ words + timings ─┐
                                │                                    ├─→ cues
                                └─→ diarization ──→ speaker turns ───┘    │
                                                                          ▼
             burned-in video ←── ffmpeg ←── ASS layout ←── LLM translation
```

1. **Transcribe** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, no PyTorch). On Apple Silicon it transparently upgrades to `mlx-whisper` for GPU speed if installed.
2. **Identify speakers** — [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) running pyannote segmentation through onnxruntime. No HuggingFace token, no licence gate, ~35 MB.
3. **Build cues** — split where the speaker changes mid-sentence, hold each line for its reading time, and let overlaps form.
4. **Translate** — local LLM with surrounding dialogue as context.
5. **Lay out** — compute vertical slots over time so overlapping lines stack.
6. **Burn in** — ffmpeg + libass, hardware-accelerated where available.

**Why ASS and not SRT?** SRT cannot position a line. The stacking behaviour is impossible to express in it. An `.srt` is still written as a convenience, but it's a lossy export.

---

## Privacy

Everything runs locally. Whisper, speaker detection, and the translation LLM all execute on your machine; no audio, transcript, or metadata is ever transmitted.

After `subgen setup`, the tool sets `HF_HUB_OFFLINE` before any model library loads and verifies required files are on disk before starting — so a normal run makes **no network calls at all**. If something is missing you get an immediate, actionable message instead of a silent hang. Verify any time with `subgen --status`.

---

## Troubleshooting

<details open>
<summary><b>"No module named _tkinter"</b> — the most common problem</summary>

Your Python was built without GUI support. This is very common with Homebrew Python and some system Pythons.

**Easiest fix** — use a `uv` Python, which bundles it:
```bash
uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e .
```

Or install the toolkit for your existing Python:
```bash
brew install python-tk          # macOS
sudo apt install python3-tk     # Debian/Ubuntu
sudo dnf install python3-tkinter # Fedora
```

You can also skip the GUI entirely and use the command line: `subgen video.mp4 --lang en`
</details>

<details>
<summary><b>"ffmpeg was not found"</b></summary>

```bash
brew install ffmpeg              # macOS
sudo apt install ffmpeg          # Debian/Ubuntu
winget install Gyan.FFmpeg       # Windows
```
</details>

<details>
<summary><b>Subtitles aren't translated</b></summary>

Run `subgen --status`. Most likely Ollama isn't running or has no models:
```bash
ollama serve          # start the daemon
ollama pull qwen3:8b  # get a model
subgen --list-models  # confirm we can see it
```
</details>

<details>
<summary><b>Lines never stack</b></summary>

Stacking needs speaker detection. Check `subgen --status` shows *Speaker models: ready*; if not, run `subgen setup`. Also make sure `--no-speakers` isn't set.

Note that stacking only happens when a *different* speaker starts while a line is still up. In a single-speaker video (a lecture, a monologue) lines correctly never stack.
</details>

<details>
<summary><b>Chinese subtitles show as boxes (□□□)</b></summary>

The font lacks CJK glyphs. Install one:
```bash
sudo apt install fonts-noto-cjk   # Linux
```
macOS and Windows ship suitable fonts already.
</details>

<details>
<summary><b>Transcription is slow</b></summary>

`large-v3` is the most accurate and the slowest. Try `--whisper-model medium` or `small`. On Apple Silicon, `uv pip install -e ".[mlx]"` enables GPU transcription for a large speedup.
</details>

---

## Development

```bash
uv pip install -e ".[dev]"
pytest
```

The test suite covers the dialogue layout rules — when lines stack, when they
replace, and how text is wrapped — and reads as a specification of the intended
on-screen behaviour.

---

## License

MIT
