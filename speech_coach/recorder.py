"""Record from the microphone until the user presses Enter."""
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16_000  # Hz
KEYPRESS_TRIM_S = 0.15 # drop the final Enter-key click from the recording

def record_until_enter(out_file: Path) -> Path:
    chunks: queue.Queue[np.ndarray] = queue.Queue()

    def on_audio(indata, frames, time, status) -> None:
        chunks.put(indata.copy())  # runs on sounddevice's audio thread

    input("\nPress Enter to start recording...\n")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", callback=on_audio):
        input("Recording...press Enter to stop.\n")

    recorded = []
    while not chunks.empty():
        recorded.append(chunks.get())
    if not recorded:
        raise SystemExit("No audio recorded. Please check your microphone and try again.")

    audio = np.concatenate(recorded)[:-int(KEYPRESS_TRIM_S * SAMPLE_RATE)]  # drop the final Enter-key click
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_file, audio, SAMPLE_RATE)
    return out_file