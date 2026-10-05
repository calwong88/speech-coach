"""Pause detection from the raw audio signal, not from word timestamps.

CrisperWhisper's timestamps run back to back, so gaps between words can't be
measured from the. Instead we measure loudness over time and find quiet stretches.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME_S = 0.03       # analyse the audio in 30 ms slices
MIN_PAUSE_S = 0.5    # a quiet stretch at least this long counts as a pause
LONG_PAUSE_S = 2.0   # long enough that a listener clearly notices
SILENCE_RATIO = 0.1  # "quiet" = below 10% of typical speaking loudness

@dataclass(frozen=True)
class Pause:
    start: float  # seconds
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_audio(wav:Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(wav, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)  # stereo -> mono
    return samples, sample_rate


def frame_loudness(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """RMS loudness of each 30 ms frame."""
    frame_len = int(FRAME_S * sample_rate)
    n_frames = len(samples) // frame_len
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames**2, axis=1))


def quiet_frames(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """True for each 30 ms frame that's quiet relative to this recording's speech level."""
    loudness = frame_loudness(samples, sample_rate)
    if loudness.size == 0:
        return np.zeros(0, dtype=bool)
    # Assumes you're speaking for most of the recording, so the 95th percentile
    # is a good estimate of your normal speaking loudness.
    speech_level = np.percentile(loudness, 95)
    if speech_level == 0:
        return np.zeros(0, dtype=bool)  # pure silence: every frame is quiet
    return loudness < speech_level * SILENCE_RATIO


def speech_end(samples: np.ndarray, sample_rate: int) -> float | None:
    """Time (s) where the last non-quiet frame ends, or None if there's no speech."""
    loud = np.flatnonzero(~quiet_frames(samples, sample_rate))
    if loud.size == 0:
        return None
    return round((loud[-1] + 1) * FRAME_S, 2)


def find_pauses(samples: np.ndarray, sample_rate: int) -> list[Pause]:
    pauses: list[Pause] = []
    run_start: int | None = None
    for i, quiet in enumerate(quiet_frames(samples, sample_rate)):
        if quiet and run_start is None:
            run_start = i  # a quiet stretch begins
        elif not quiet and run_start is not None:
            # A quiet stretch just ended. Keep it only if it's long enough
            # and didn't start at frame 0 (that's silence before you spoke).
            if run_start > 0 and (i - run_start) * FRAME_S >= MIN_PAUSE_S:
                pauses.append(
                    Pause(start=round(run_start * FRAME_S, 2), end=round(i * FRAME_S, 2))
                )
            run_start = None
    # A quiet stretch still open here is trailing silence, not a pause.
    return pauses
