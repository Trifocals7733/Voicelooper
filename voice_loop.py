"""Local voice loop: microphone -> Silero VAD -> Whisper STT -> Kokoro TTS.

The whole pipeline runs locally. Each spoken utterance follows this path:

    1. The sounddevice stream feeds fixed-size mic chunks into `on_chunk`.
    2. Silero VAD scores every chunk; speech-chunks are buffered.
    3. When SILENCE_DURATION passes without speech, the buffer becomes an
       utterance and is transcribed by faster-whisper.
    4. The transcript is fed back through Kokoro TTS and played on all
       configured output devices (echo-loop latency benchmark).

Echo prevention: while TTS audio is playing, the mic callback drops all
chunks (`UtteranceCollector.muted`) so the assistant never hears itself.
"""

import os
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from kokoro import KPipeline
from silero_vad import load_silero_vad

# ---------------------------------------------------------------------------
# Console styling
#
# Plain ANSI codes, no dependencies. The classic `os.system("")` call below
# enables VT escape-sequence processing on legacy Windows terminals.
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
#
# Every tunable lives here, grouped by subsystem so each block reads on its own.
# ---------------------------------------------------------------------------

# --- Microphone capture & VAD -------------------------------------------------
SAMPLE_RATE = 16_000        # Hz - both Whisper and Silero VAD expect 16 kHz mono.
CHUNK_SIZE = 512            # Samples per mic callback (~32 ms at 16 kHz),
                            # which matches what Silero VAD was trained on.
SILENCE_DURATION = 0.20     # Seconds of trailing silence that close an utterance.
                            # Raised slightly so short mid-sentence pauses are kept.
MIN_SPEECH_DURATION = 0.25  # Ignore buffers shorter than this (coughs, clicks).
VAD_SPEECH_THRESHOLD = 0.5  # Probability above which a chunk counts as speech.

# --- Speaker playback ---------------------------------------------------------
CHUNK_WRITE_SIZE = 1024     # Samples written per stream.write() call - smaller
                            # writes mean smoother playback under load.

# --- Output devices -----------------------------------------------------------
# Ordered fallback list: substring of the device name + host API. All matched
# devices are played simultaneously (e.g. speakers AND virtual cable).
TARGET_OUTPUTS = [
    ("Microsoft Sound Mapper - Output", "MME"),
    ("CABLE Input", "MME"),
]

# --- Kokoro TTS -----------------------------------------------------------------
TTS_LANG_CODE = "a"         # Kokoro's American-English language pack.
TTS_VOICE = "am_adam"
TTS_SPEED = 1.0
TTS_SAMPLE_RATE = 24_000    # Hz - native sample rate of Kokoro output audio.
TTS_VOLUME = 0.3            # Output gain multiplier.

# --- Whisper STT ------------------------------------------------------------------
WHISPER_MODEL_NAME = "moondream/parakeet-redux"
WHISPER_LANGUAGE = "en"


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------

def get_output_device_id(name_substring: str, host_api: str = "MME") -> int | None:
    """Return the sounddevice index of the first matching output device.

    Matches are case-insensitive on both the host-API name (e.g. MME,
    WASAPI) and the device name substring. Returns None when nothing fits.
    """
    # Map host-api index -> readable name so we can filter by API family.
    host_apis = {i: api["name"] for i, api in enumerate(sd.query_hostapis())}

    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] == 0:      # Skip pure input devices.
            continue
        dev_api = host_apis.get(dev["hostapi"], "")
        if (
            host_api.lower() in dev_api.lower()
            and name_substring.lower() in dev["name"].lower()
        ):
            return idx
    return None


def resolve_output_devices() -> list[int]:
    """Resolve TARGET_OUTPUTS to concrete device indices.

    Warns (rather than crashes) for missing devices - only one entry is
    usually present per machine.
    """
    devices: list[int] = []
    all_devices = sd.query_devices()
    for name, api in TARGET_OUTPUTS:
        dev_id = get_output_device_id(name, api)
        if dev_id is None:
            print(f"{YELLOW}Warning:{RESET} could not find audio device '{name}' ({api})")
        else:
            devices.append(dev_id)
            print(f"Output: {all_devices[dev_id]['name']} {DIM}(#{dev_id}, {api}){RESET}")
    if not devices:
        print(f"{YELLOW}No configured output devices found.{RESET}")
    return devices


# ---------------------------------------------------------------------------
# Model loading
#
# Each loader only constructs its model; nothing model-related happens at
# import time, so `main()` controls exactly when the slow initializations run.
# ---------------------------------------------------------------------------

def compute_device() -> tuple[str, str]:
    """Pick torch device and the matching faster-whisper quantization."""
    has_cuda = torch.cuda.is_available()
    return (
        "cuda" if has_cuda else "cpu",
        "float16" if has_cuda else "int8",
    )


def load_step(label: str, fn):
    """Run one slow init step with a 'Label...' -> 'Label ok' progress line.

    Centralized so every loader gets identical, consistent feedback and the
    loader functions themselves stay pure construction logic.
    """
    print(f"{DIM}[init]{RESET} {label}...", flush=True)
    result = fn()
    # Overwrite the pending line with a green check.
    sys.stdout.write("\r\x1b[2K")
    print(f"{DIM}[init]{RESET} {label} {GREEN}ok{RESET}")
    return result


def load_stt() -> WhisperModel:
    """Construct the faster-whisper transcription model (no logging here;
    progress feedback is handled by `load_step` in `main`)."""
    device, compute_type = compute_device()
    return WhisperModel(WHISPER_MODEL_NAME, device=device, compute_type=compute_type)


def load_tts() -> KPipeline:
    """Load the Kokoro TTS pipeline and warm it up once.

    The warmup run pays JIT/graph-init cost up front so the first real
    utterance does not pay the multi-second cold-start penalty.
    """
    tts = KPipeline(lang_code=TTS_LANG_CODE, device=compute_device()[0])
    # Consume every yielded chunk before reporting ready.
    for _ in tts("Hello.", voice=TTS_VOICE, speed=TTS_SPEED):
        pass
    return tts


def is_speech(vad_model, chunk: np.ndarray) -> bool:
    """Score one float32 chunk; True when speech probability exceeds threshold."""
    tensor = torch.from_numpy(chunk).float()
    with torch.no_grad():
        probability = vad_model(tensor, SAMPLE_RATE).item()
    return probability > VAD_SPEECH_THRESHOLD


# ---------------------------------------------------------------------------
# Mic capture & utterance segmentation
# ---------------------------------------------------------------------------

class UtteranceCollector:
    """Turns raw mic chunks into finished utterances via a tiny state machine.

    States:
      idle         - discarding silence, waiting for the first speech chunk.
      listening    - buffering audio while speech continues and for a short
                     grace period afterwards (pauses inside sentences).

    `muted` short-circuits everything while TTS is playing so the speaker
    output can never feed back into recognition.
    """

    def __init__(self, vad_model):
        self._vad = vad_model
        self.reset()

    def reset(self):
        self.muted = False             # Set by speak(); blocks mic capture.
        self.speech_started = False    # Idle vs. listening flag.
        self.last_speech_time = None   # time.time() of last detected speech.
        self.buffer: list[np.ndarray] = []

    def on_chunk(self, chunk: np.ndarray) -> None:
        """Feed one captured chunk into the state machine.

        Expected shape: 1-D float32 samples. The sounddevice callback
        (see `on_audio`) strips the channel dimension before calling this.
        """
        if self.muted:
            return

        if is_speech(self._vad, chunk):
            if not self.speech_started:
                self.speech_started = True
                print(f"\n{CYAN}Listening...{RESET}")
            self.last_speech_time = time.time()
            self.buffer.append(chunk)
        elif self.speech_started:
            # Trailing non-speech is still buffered so word endings are not
            # chopped off before the silence timeout closes the utterance.
            self.buffer.append(chunk)

    def pop_if_finished(self) -> np.ndarray | None:
        """Return concatenated audio when the utterance has fully ended.

        An utterance counts as ended once SILENCE_DURATION elapsed without
        any detected speech. Consumed audio (and all state) is then reset;
        returns None whenever no complete utterance is available yet.
        """
        if not self.speech_started or self.last_speech_time is None:
            return None
        if time.time() - self.last_speech_time < SILENCE_DURATION:
            return None

        # Silence timeout hit: the speech segment is complete.
        audio = np.concatenate(self.buffer) if self.buffer else None
        self.reset()
        return audio

    def on_audio(self, indata, frames, time_info, status) -> None:
        """sounddevice InputStream callback wrapper for `on_chunk`.

        Copies the mono channel out of the interleaved input buffer so we own
        the array that is appended to `buffer` (the callback's underlying
        storage may be reused by portaudio after we return).
        """
        self.on_chunk(indata[:, 0].copy())


# ---------------------------------------------------------------------------
# TTS playback
# ---------------------------------------------------------------------------

def speak(tts: KPipeline, devices: list[int], text: str) -> tuple[float | None, float]:
    """Stream Kokoro audio for `text` to every configured output device.

    Runs generation and playback concurrently:
      * a worker thread pushes synthesized chunks into a queue,
      * this thread drains the queue and blocks-writes to all streams.

    Keeping the blocking write on the caller thread guarantees we return
    only when the audio has actually been played (not merely generated).

    Returns (time_to_first_audio_ms or None, full_playback_ms) so `main`
    can render one consolidated metrics line instead of mid-stream noise.
    """
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    generation_done = threading.Event()

    t_start = time.perf_counter()
    first_audio_time: float | None = None

    def generate():
        nonlocal first_audio_time
        generator = tts(text, voice=TTS_VOICE, speed=TTS_SPEED)

        # Kokoro yields (graph_index, graph_text, audio) triples; we only use
        # the audio chunk and normalize tensors -> float32 samples.
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
        # One stream per target device; they share the same queue and are
        # written sequentially, so their outputs stay roughly synchronized.
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
                # Producer finished and drained -> playback complete.
                if generation_done.is_set():
                    break
                continue

            for i in range(0, len(audio_chunk), CHUNK_WRITE_SIZE):
                sub_chunk = audio_chunk[i : i + CHUNK_WRITE_SIZE]
                for stream in streams:
                    stream.write(sub_chunk)

    finally:
        # Always tear down streams and wait for the producer, even on error,
        # so no thread survives and holds the audio devices hostage.
        for stream in streams:
            stream.stop()
            stream.close()
        gen_thread.join()

    return (
        None if first_audio_time is None else (first_audio_time - t_start) * 1000,
        (time.perf_counter() - t_start) * 1000,
    )


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

def transcribe(stt: WhisperModel, audio: np.ndarray) -> str:
    """Run one pass of Whisper on a single utterance and return plain text."""
    segments, _info = stt.transcribe(
        audio,
        language=WHISPER_LANGUAGE,
        beam_size=1,                        # Fastest decoding - latency over accuracy.
        best_of=1,
        temperature=0,                      # Deterministic: stable echo benchmarks.
        vad_filter=False,                   # Already VAD-gated upstream.
        condition_on_previous_text=False,   # No cross-utterance context drift.
    )
    return "".join(segment.text for segment in segments).strip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    device, _ = compute_device()
    print(f"""
{BOLD}  STT -> TTS Echo{RESET} {DIM}| {WHISPER_MODEL_NAME} · Kokoro '{TTS_VOICE}' · {device.upper()}{RESET}
  Speak normally. {DIM}Ctrl+C to quit.{RESET}
""")

    output_devices = resolve_output_devices()
    stt = load_step(f"Whisper ({WHISPER_MODEL_NAME})", load_stt)
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
                time.sleep(0.02)               # Cheap wake/poll cadence.
                audio = mic.pop_if_finished()
                if audio is None:
                    continue

                duration = len(audio) / SAMPLE_RATE
                if duration < MIN_SPEECH_DURATION:
                    continue                   # Too short to contain a sentence.

                roundtrip_start = time.perf_counter()

                # --- STT ---
                # 'Transcribing...' on its own line, erased once the text is in
                # so the transcript block starts clean.
                sys.stdout.write(f"{DIM}Transcribing...{RESET}\r")
                stt_start = time.perf_counter()
                text = transcribe(stt, audio)
                stt_ms = time.perf_counter() - stt_start
                if not text:
                    # Clear the pending line and say so instead of going quiet.
                    sys.stdout.write("\x1b[2K\r")
                    print(f"{YELLOW}(nothing recognized){RESET}")
                    continue

                speech_secs = len(audio) / SAMPLE_RATE
                sys.stdout.write("\x1b[2K\r")
                print(f"{BOLD}{GREEN}YOU:{RESET} {text}")

                # --- TTS ---
                # Mute the mic during playback so the speaker output never loops
                # back through recognition, unmute right after it finishes.
                mic.muted = True
                ttfa_ms, tts_ms = speak(tts, output_devices, text)
                mic.muted = False

                total_ms = time.perf_counter() - roundtrip_start
                ttfa_txt = "n/a" if ttfa_ms is None else f"{ttfa_ms:.0f}ms"
                print(
                    f"{DIM}stt {stt_ms*1000:.0f}ms"
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
