"""Smoke test: record from the mic. transcribe on the GPU, print what Whisper heard."""
import os
import sys
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf

# Windows only: tell Python where pip put the CUDE DLLs.
if sys.platform == "win32":
    nvidia_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for lib in ("cublas", "cudnn"):
        dll_dir = str((nvidia_dir / lib / "bin"))
        os.add_dll_directory(dll_dir)                                   # for Python's own loader
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]  # for CTranslate2's loader

from faster_whisper import WhisperModel # must come AFTER the DLL paths are added

SAMPLE_RATE = 16_000 # Whisper works on 16 kHz audio
SECONDS = 10
OUT_FILE = Path("data/smoke_test.wav")


def record(seconds: int) -> Path:
    print(f"Recording for {seconds}s... talk naturally, ums and all.")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait() # block until recording finishes
    OUT_FILE.parent.mkdir(exist_ok=True)
    sf.write(OUT_FILE, audio, SAMPLE_RATE)
    return OUT_FILE


def main() -> None:
    wav = record(SECONDS)
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    start = time.perf_counter()
    segments, _info = model.transcribe(str(wav), word_timestamps=True)
    for segment in segments: # transcription actually runs her, as you loop
        for word in segment.words:
            print(f"{word.start:6.2f}s {word.word}")
    print(f"\nTranscribed in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()