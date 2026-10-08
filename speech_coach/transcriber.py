"""Speech-to-text: turn a WAV file into a list of timed words."""
import os
import sys
from pathlib import Path

from speech_coach.words import Word, split_token

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


class Transcriber:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model = WhisperModel(model_name, device="cuda", compute_type="float16")

    def transcribe(self, wav: Path, **overrides) -> list[Word]:
        options = {
            "language": "en",
            "word_timestamps": True,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
            "condition_on_previous_text": False,  # prevents repetition loops (see experiments/loop_test.py)
            **overrides,  # later keys win, so an experiement can change any default
        }
        segments, _info = self._model.transcribe(str(wav), **options)
        words: list[Word] = []
        for segment in segments:
            for token in segment.words:
                # Pieces split from one token share its timestamps (e.g. "um" + "I").
                for text in split_token(token.word):
                    words.append(
                        Word(text=text, start=token.start, end=float(token.end))
                    )
        return words


if __name__ == "__main__":
    for word in Transcriber().transcribe(Path("data/smoke_test.wav")):
        print(f"{word.start:6.2f}-{word.end:6.2f}s {word.text!r}")