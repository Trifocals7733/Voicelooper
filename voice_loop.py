"""Local voice loop: microphone -> Silero VAD -> Parakeet-Redux STT -> Kokoro TTS.

Pipeline overview:
    1. sounddevice feeds fixed-size audio chunks into UtteranceCollector.
    2. Silero VAD continuously scores chunks; rolling pre-roll buffer prevents
       chopping off initial syllables/words.
    3. Trailing silence (SILENCE_DURATION) marks the utterance boundary.
    4. moondream/parakeet-redux (Photon) transcribes the speech buffer.
    5. Kokoro TTS synthesizes the response and streams audio to configured outputs.
    6. Mic input is muted during playback to eliminate acoustic feedback.
"""

from collections import deque
import os
import queue
import sys
import tempfile
import threading
import time

from kokoro import KPipeline
import moondream as md
import numpy as np
from silero_vad import load_silero_vad
import sounddevice as sd
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Console styling
# ---------------------------------------------------------------------------
os.system("")

RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# --- Microphone capture & VAD -----------------------------------------------
SAMPLE_RATE = 16_000  # Hz (Parakeet & Silero VAD standard)
CHUNK_SIZE = 512  # Samples per chunk (~32 ms at 16 kHz)
SILENCE_DURATION = 0.40  # Seconds of silence before finalizing sentence
MIN_SPEECH_DURATION = 0.30  # Ignore noise shorter than this (coughs, clicks)
VAD_SPEECH_THRESHOLD = 0.45  # Probability threshold for active speech
PRE_ROLL_MS = 300  # Audio buffered prior to detection (prevents cut-off)
CHUNKS_TO_KEEP = int(
    (PRE_ROLL_MS / 1000) / (CHUNK_SIZE / SAMPLE_RATE)
)  # ~9 chunks

# --- Speaker playback -------------------------------------------------------
CHUNK_WRITE_SIZE = 1024  # Samples per stream write

# --- Output devices ---------------------------------------------------------
TARGET_OUTPUTS = [
    ("Microsoft Sound Mapper - Output", "MME"),
    ("CABLE Input", "MME"),
]

# --- Kokoro TTS -------------------------------------------------------------
TTS_LANG_CODE = "a"  # American English
TTS_VOICE = "am_adam"
TTS_SPEED = 1.1
TTS_SAMPLE_RATE = 24_000  # Hz
TTS_VOLUME = 0.3

# --- STT (Parakeet Redux) ---------------------------------------------------
STT_MODEL_NAME = "moondream/parakeet-redux"
# Välj "cuda" för GPU eller "cpu" om du vill spara all GPU-kapacitet till Kokoro/LLM
STT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------


def get_output_device_id(
    name_substring: str, host_api: str = "MME"
) -> int | None:
    host_apis = {i: api["name"] for i, api in enumerate(sd.query_hostapis())}

    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] == 0:
            continue
        dev_api = host_apis.get(dev["hostapi"], "")
        if (
            host_api.lower() in dev_api.lower()
            and name_substring.lower() in dev["name"].lower()
        ):
            return idx
    return None


def resolve_output_devices() -> list[int]:
    devices: list[int] = []
    all_devices = sd.query_devices()
    for name, api in TARGET_OUTPUTS:
        dev_id = get_output_device_id(name, api)
        if dev_id is None:
            print(
                f"{YELLOW}Warning:{RESET} could not find audio device '{name}' ({api})"
            )
        else:
            devices.append(dev_id)
            print(
                f"Output: {all_devices[dev_id]['name']} {DIM}(#{dev_id}, {api}){RESET}"
            )
    if not devices:
        print(f"{YELLOW}No configured output devices found.{RESET}")
    return devices


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def compute_device() -> tuple[str, str]:
    has_cuda = torch.cuda.is_available()
    return (
        "cuda" if has_cuda else "cpu",
        "float16" if has_cuda else "int8",
    )


def load_step(label: str, fn):
    print(f"{DIM}[init]{RESET} {label}...", flush=True)
    result = fn()
    sys.stdout.write("\r\x1b[2K")
    print(f"{DIM}[init]{RESET} {label} {GREEN}ok{RESET}")
    return result


def load_stt():
    # Photon initierar Parakeet-Redux optimerat för vald hårdvara
    return md.photon(STT_MODEL_NAME, device=STT_DEVICE)


def load_tts() -> KPipeline:
    tts = KPipeline(lang_code=TTS_LANG_CODE, device=compute_device()[0])
    # Warmup
    for _ in tts("Hello.", voice=TTS_VOICE, speed=TTS_SPEED):
        pass
    return tts


def is_speech(vad_model, chunk: np.ndarray) -> bool:
    tensor = torch.from_numpy(chunk).float()
    with torch.no_grad():
        probability = vad_model(tensor, SAMPLE_RATE).item()
    return probability > VAD_SPEECH_THRESHOLD


# ---------------------------------------------------------------------------
# Mic capture & utterance segmentation
# ---------------------------------------------------------------------------


class UtteranceCollector:

    def __init__(self, vad_model):
        self._vad = vad_model
        self.pre_buffer = deque(maxlen=CHUNKS_TO_KEEP)
        self.reset()

    def reset(self):
        self.muted = False
        self.speech_started = False
        self.last_speech_time = None
        self.buffer: list[np.ndarray] = []
        self.pre_buffer.clear()

    def on_chunk(self, chunk: np.ndarray) -> None:
        if self.muted:
            return

        if is_speech(self._vad, chunk):
            if not self.speech_started:
                self.speech_started = True
                print(f"\n{CYAN}Listening...{RESET}")
                # Prepend the pre-roll audio history so first syllables are intact
                self.buffer.extend(list(self.pre_buffer))

            self.last_speech_time = time.time()
            self.buffer.append(chunk)
        elif self.speech_started:
            self.buffer.append(chunk)
        else:
            self.pre_buffer.append(chunk)

    def pop_if_finished(self) -> np.ndarray | None:
        if not self.speech_started or self.last_speech_time is None:
            return None
        if time.time() - self.last_speech_time < SILENCE_DURATION:
            return None

        audio = np.concatenate(self.buffer) if self.buffer else None
        self.reset()
        return audio

    def on_audio(self, indata, frames, time_info, status) -> None:
        self.on_chunk(indata[:, 0].copy())


# ---------------------------------------------------------------------------
# TTS playback
# ---------------------------------------------------------------------------


def speak(
    tts: KPipeline, devices: list[int], text: str
) -> tuple[float | None, float]:
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    generation_done = threading.Event()

    t_start = time.perf_counter()
    first_audio_time: float | None = None

    def generate():
        nonlocal first_audio_time
        generator = tts(text, voice=TTS_VOICE, speed=TTS_SPEED)

        for _, _, audio in generator:
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()

            audio = (audio * TTS_VOLUME).astype(np.float32)

            if first_audio_time is None:
                first_audio_time = time.perf_counter()

            audio_queue.put(audio)

        generation_done.set()

    gen_thread = threading.Thread(target=generate, daemon=True)
    gen_thread.start()

    streams: list[sd.OutputStream] = []
    try:
        for dev in devices:
            stream = sd.OutputStream(
                samplerate=TTS_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=dev,
            )
            stream.start()
            streams.append(stream)

        while True:
            try:
                audio_chunk = audio_queue.get(timeout=0.05)
            except queue.Empty:
                if generation_done.is_set():
                    break
                continue

            for i in range(0, len(audio_chunk), CHUNK_WRITE_SIZE):
                sub_chunk = audio_chunk[i : i + CHUNK_WRITE_SIZE]
                for stream in streams:
                    stream.write(sub_chunk)

    finally:
        for stream in streams:
            stream.stop()
            stream.close()
        gen_thread.join()

    ttfa = (
        None
        if first_audio_time is None
        else (first_audio_time - t_start) * 1000
    )
    total_tts_ms = (time.perf_counter() - t_start) * 1000
    return ttfa, total_tts_ms


# ---------------------------------------------------------------------------
# STT (Parakeet Redux transcription via Photon)
# ---------------------------------------------------------------------------


def transcribe(stt_engine, audio: np.ndarray) -> str:
    # Skriver segmentet till en temporär WAV-fil för Photon
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name

    try:
        sf.write(tmp_name, audio, SAMPLE_RATE)
        result = stt_engine.transcribe(audio=tmp_name)
        text = result.get("text", "").strip()
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    return text


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"""
{BOLD}  STT -> TTS Echo{RESET} {DIM}| {STT_MODEL_NAME} · Kokoro '{TTS_VOICE}' · {STT_DEVICE.upper()}{RESET}
  Speak normally. {DIM}Ctrl+C to quit.{RESET}
""")

    output_devices = resolve_output_devices()
    stt = load_step(f"Parakeet Redux ({STT_MODEL_NAME})", load_stt)
    tts = load_step("Kokoro TTS + warmup", load_tts)
    vad_model = load_step("Silero VAD", load_silero_vad)
    mic = UtteranceCollector(vad_model)

    print(f"\n{GREEN}Ready.{RESET} Waiting for speech...\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=mic.on_audio,
        ):
            while True:
                time.sleep(0.02)
                audio = mic.pop_if_finished()
                if audio is None:
                    continue

                duration = len(audio) / SAMPLE_RATE
                if duration < MIN_SPEECH_DURATION:
                    continue

                roundtrip_start = time.perf_counter()

                # --- STT ---
                sys.stdout.write(f"{DIM}Transcribing...{RESET}\r")
                stt_start = time.perf_counter()
                text = transcribe(stt, audio)
                stt_ms = (time.perf_counter() - stt_start) * 1000

                if not text:
                    sys.stdout.write("\x1b[2K\r")
                    print(f"{YELLOW}(nothing recognized){RESET}")
                    continue

                speech_secs = len(audio) / SAMPLE_RATE
                sys.stdout.write("\x1b[2K\r")
                print(f"{BOLD}{GREEN}YOU:{RESET} {text}")

                # --- TTS ---
                mic.muted = True
                ttfa_ms, tts_ms = speak(tts, output_devices, text)
                mic.muted = False

                total_ms = (time.perf_counter() - roundtrip_start) * 1000
                ttfa_txt = "n/a" if ttfa_ms is None else f"{ttfa_ms:.0f}ms"
                print(
                    f"{DIM}stt {stt_ms:.0f}ms"
                    f" | ttfa {ttfa_txt}"
                    f" | tts {tts_ms:.0f}ms"
                    f" | total {total_ms:.0f}ms"
                    f" | {speech_secs:.2f}s heard{RESET}"
                )
                print()

    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    main()