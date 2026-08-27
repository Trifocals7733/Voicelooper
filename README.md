<div align="center">

# 🎙️ Live Speech → Text → Speech

**A fully client-side voice loop — Silero VAD → Whisper STT → Kokoro TTS — running in your browser via WebGPU.**

*No server inference. No cloud API. Your voice never leaves your machine.*

[![Runs in browser](https://img.shields.io/badge/runs_in-browser_webGPU-blue)](#-quick-start-web)
[![STT](https://img.shields.io/badge/STT-Whisper--base_(ONNX)-4f8cff)](https://huggingface.co/onnx-community/whisper-base)
[![TTS](https://img.shields.io/badge/TTS-Kokoro--82M-2de0a7)](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<!-- TODO before publishing: drop a screenshot / screen-recording here.
     A GIF of the app speaking after you speak sells this project instantly:
     <img src="docs/demo.gif" alt="demo" width="460"> -->

</div>

---

## ✨ Features

- **100% local** — all three models run inside the browser tab on WebGPU (WASM fallback included); zero network calls at runtime except model download
- **Barge-in** — start talking mid-answer and playback stops instantly while work in flight is discarded
- **Streaming TTS** — audio is synthesized sentence-by-sentence; measured metric is *time-to-first-audio*, not total generation time
- **Live latency dashboard** — STT / TTFA / TTS / TOTAL per round-trip
- **In-page settings** — voice catalog (54 voices), speed, volume, end-of-speech silence window
- **No LLM, no database, no auth** — an echo benchmark, nothing more

## 🚀 Quick Start (Web)

```bash
git clone https://github.com/Trifocals7733/Voicelooper.git
cd voicelooper/web
python serve.py        # Python 3.x, stdlib only — hosts static files, does NO inference
```

Open **http://127.0.0.1:8000** in Chrome or Edge → click **▶** → allow microphone.

| | |
|---|---|
| **First load** | downloads ~500 MB of models from HuggingFace (Whisper base + Kokoro-82M fp32); cached by the browser afterwards |
| **Requirements** | any WebGPU-capable Chromium browser (falls back to slower WASM automatically) |
| **Recommended** | headphones 🎧 — otherwise the mic hears the TTS playback |

> Why the Python server at all? It sets the `COOP`/`COEP` headers required for
> `SharedArrayBuffer`, which the threaded WASM fallback needs.

## 💻 Quick Start (CLI)

The same loop as a native terminal app — bigger models, lower latency:

```bash
pip install numpy sounddevice torch faster-whisper kokoro silero-vad
python voice_loop.py
```

Uses `distil-large-v3` Whisper (CUDA if available) + Kokoro TTS, echoes your
speech back, and prints per-round-trip timings:

```
● Listening...
YOU: hello world
stt 245ms | ttfa 180ms | tts 620ms | total 1045ms | 2.30s heard
```

## 🏗️ Architecture

```
Web (everything inside the tab)                 CLI (voice_loop.py)
────────────────────────────                    ────────────────────
@ricky0123/vad-web ─ Silero VAD                 Silero VAD (utterance gating)
      │                                               │
Transformers.js ─── Whisper STT                 faster-whisper distil-large-v3
      │                                               │
kokoro-js ───────── Kokoro TTS                  Kokoro KPipeline (threaded gen,
      │                                         streaming playback to N devices)
AudioContext playback                                 │
barge-in via runId invalidation                 stdout latency report
```

## ⚙️ Configuration

CLI tunables live at the top of [`voice_loop.py`](voice_loop.py):

| Constant | Default | Meaning |
|---|---|---|
| `SILENCE_DURATION` | `0.20 s` | trailing silence that closes an utterance |
| `MIN_SPEECH_DURATION` | `0.25 s` | shorter buffers are discarded |
| `TARGET_OUTPUTS` | Sound Mapper + CABLE Input | output devices played simultaneously |
| `TTS_VOICE` / `TTS_SPEED` | `am_adam` / `1.0` | Kokoro voice & rate |
| `WHISPER_MODEL_NAME` | `distil-large-v3` | swap freely (CPU users: try `base` or `small`) |

Web settings (voice, speed, volume, silence) live in the collapsible panel — no reload needed.

## 📊 Metrics explained

- **STT** — transcription time for the full utterance
- **TTFA** — time from TTS start until the first audible chunk (the number that matters for perceived latency)
- **TTS** — total time from synthesis start to playback finished
- **TOTAL** — end of speech detection → speaker finishes playing

## ⚠️ Known trade-offs

- Browser Whisper is `whisper-base` (74M params) — `distil-large-v3` can't run in a tab, so web STT is less accurate than the CLI
- TTS audio is resampled 24 kHz → AudioContext rate in JS (linear interpolation)
- No VAD-based barge-in during *transcription* — only during playback

## 📁 Project layout

```
web/
  index.html    # entire web app (UI + logic, heavily commented)
  serve.py      # static file server with COOP/COEP headers
voice_loop.py   # standalone CLI loop
```

## 🙏 Credits & licenses

This project only orchestrates these models at runtime — nothing is
redistributed in this repository. Still, credit where it's due:

| Component | Role | License | Source |
|---|---|---|---|
| Silero VAD | speech gating | code: **MIT**, model weights: **CC BY-NC-SA 4.0** ⚠️ | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) · [@ricky0123/vad-web](https://github.com/ricky0123/vad) |
| Whisper | STT | **MIT** | [openai/whisper](https://github.com/openai/whisper) · [onnx-community/whisper-base](https://huggingface.co/onnx-community/whisper-base) |
| Kokoro | TTS | **Apache 2.0** | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) · [kokoro-js](https://github.com/hexgrad/kokoro) |
| faster-whisper | CLI STT runtime | **MIT** | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| Transformers.js | web inference runtime | **Apache 2.0** | [huggingface/transformers.js](https://github.com/huggingface/transformers.js) |
| onnxruntime-web | execution backend | **MIT** | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |

> ⚠️ **Silero VAD model weights are non-commercial** (`CC BY-NC-SA 4.0`),
> despite the MIT library code. If this project ever goes commercial,
> swap VAD or obtain a license from the Silero Team.

## License

Released under the [MIT License](LICENSE) — © 2026 Trifocals7733.
Note that this covers *this repository's code only*; the models it runs are
licensed separately (see [Credits & licenses](#-credits--licenses) — in
particular, Silero VAD weights remain non-commercial).
