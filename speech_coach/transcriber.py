"""Speech-to-text: turn a WAV file into a list of timed words."""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

MODEL_NAME = "nyrahealth/faster_CrisperWhisper"


def _add_cuda_dlls() -> None:
    """Windows only: make pip's CUDA DLLs findable before faster-whisper loads."""
    if sys.platform != "win32":
        return
    nvidia_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for lib in ("cublas", "cudnn"):
        dll_dir = str((nvidia_dir / lib / "bin"))
        os.add_dll_directory(dll_dir)                                   # for Python's own loader
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]  # for CTranslate2's loader


_add_cuda_dlls()

from faster_whisper import WhisperModel # noqa: E402 (must come after the DLL setup)


@dataclass(frozen=True)
class Word:
    text: str
    start: float # seconds from the start of the recording
    end: float


class Transcriber:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model = WhisperModel(model_name, device="cuda", compute_type="float16")

    def transcribe(self, wav: Path) -> list[Word]:
        segments, _info = self._model.transcribe(
            str(wav), language="en", word_timestamps=True
        )
        return [
            Word(text=w.word.strip(), start=w.start, end=w.end)
            for segment in segments
            for w in segment.words
        ]


if __name__ == "__main__":
    for word in Transcriber().transcribe(Path(Path("data/smoke_test.wav"))):
        print(f"{word.start:6.2f}-{word.end:6.2f}s {word.text!r}")