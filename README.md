# Speech Coach

A local, privacy-first speech practice tool. Speak an answer into your microphone, and Speech Coach measures **how** you said it — filler words, pace, pauses, and repetitions — so you can practice interviews and everyday conversation and track real improvement over time.

Everything runs on your own machine. Your voice never leaves it.

## Why I built this

I wanted a private way to practice speaking — interview answers, explaining technical ideas, everyday conversation — and get honest, measurable feedback instead of guessing how I sounded. Building it myself was also a way to learn Python, local AI models, and test-driven design along the way.

## What it does (Phase 1)

Run one command, answer a question out loud, press Enter, and get a scorecard:

```
=== Transcript ===
So um I think the main reason I moved into identity and access management is ...

=== Scorecard ===
Duration:         61.2 s
Pace:             118 words/min (120 words)
Time pausing:     19.4 s of 61.2 s
Speaking rate:    172 words/min (pauses removed)
Fillers:          4 (3.9/min) {'um': 1, 'uh': 3}
Possible fillers: {'like': 1}
Repetitions:      i
Pauses >= 0.5 s:  14 (long >= 2 s: 1)
```
*(Example output.)*

| Metric | What it tells you |
|---|---|
| **Fillers** (um, uh) | The clearest "nervous speaker" signal, reported as a rate per minute so short and long answers compare fairly |
| **Possible fillers** (like, you know, basically) | Flagged but never scored — "I'd like to" is a real word, so context decides |
| **Pace vs. speaking rate** | Pace includes pauses; speaking rate removes them. Separating the two shows whether you talk fast, pause a lot, or both |
| **Pauses** | Measured from the audio itself; long pauses (2 s+) are the ones listeners notice |
| **Repetitions** | Restarts like "I'm, I'm" — moments where a thought got ahead of the words |

## How it works

```
microphone ──► recorder ──► transcriber ──► cleanup ──► metrics engine ──► scorecard
  (Enter to     (sounddevice)  (CrisperWhisper    (silence check,  (fillers, pace,
   start/stop)                  on the GPU)        loop filter)     repetitions)
                    │                                                     ▲
                    └──────────► pause detector (audio loudness) ─────────┘
```

| Module | Responsibility |
|---|---|
| `recorder.py` | Records until you press Enter; trims the keypress click |
| `transcriber.py` | Runs CrisperWhisper via faster-whisper; the only file that knows about the model |
| `words.py` | The `Word` type, token cleanup, silence check, and repetition-loop filter |
| `metrics.py` | Deterministic speech metrics from a list of words |
| `pauses.py` | Pause detection and end-of-speech detection from the raw audio |
| `corrections.py` | Fixes known mishearings of my vocabulary in the displayed transcript |
| `cli.py` | Connects the modules and prints the scorecard — no logic of its own |

## Design decisions

Each decision below was tested, not assumed. The experiments that justified them are in [`experiments/`](experiments/).

1. **Runs locally.** Voice recordings are personal data. Transcription runs on a local GPU, recordings are stored in `data/`, and `data/` is excluded from Git.

2. **CrisperWhisper instead of standard Whisper.** Standard Whisper is trained to write what you *meant*, so it silently drops "um", "uh", and stutters — exactly what this tool needs to measure. The well-known prompt trick didn't bring them back. CrisperWhisper, a verbatim fine-tune of Whisper, kept every filler and the repeated word. ([`filler_test.py`](experiments/filler_test.py))

3. **No AI in the scoring.** Metrics are plain, deterministic Python: the same transcript always produces the same numbers, so progress can be tracked and every rule is unit-tested. AI is reserved for interpreting results (Phase 2), never for producing them.

4. **Certain vs. possible fillers.** "Um" is always a filler; "like" often isn't. Only certain fillers count toward the score.

5. **Pauses come from the audio, not the timestamps.** Word timestamps from faster-whisper run back to back, with silence absorbed into word durations. Pauses are instead measured from loudness in 30 ms frames, with a threshold relative to each recording's own speaking level, so it adapts to any microphone.

6. **Two independent defences against hallucination.**
   - *Invented words at the end:* Whisper-family models invent text in silence. Any word that starts without at least 0.1 s of sound after it — checked against the audio — is dropped.
   - *Repetition loops:* the model once looped "A S A S…" sixty times. Disabling conditioning on previous text prevented it ([`loop_test.py`](experiments/loop_test.py)), and a filter drops any 1–3 word pattern repeated 5+ times as a safety net.
   
   Every removed word is printed, so no data disappears silently.

7. **Vocabulary fixes apply to the display only.** Giving the model a vocabulary hint backfired: `hotwords` dropped two-thirds of a recording, and `initial_prompt` cost 20% of detected fillers ([`vocab_test.py`](experiments/vocab_test.py)). Instead, a lookup table corrects known mishearings ("Bemo" → "BMO", "a m l" → "AML") in the transcript you read. Metrics always use the original words.

## Setup

**Tested on:** Windows 11, Python 3.12, NVIDIA RTX 5060 Ti (16 GB).

```powershell
git clone https://github.com/calwong88/speech-coach.git
cd speech-coach
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run downloads the CrisperWhisper model (a few GB) from Hugging Face.

**Notes for NVIDIA RTX 50-series (Blackwell) GPUs:**
- The model runs in `float16`. `int8` can crash with `CUBLAS_STATUS_NOT_SUPPORTED` on this architecture.
- On Windows, `transcriber.py` adds the pip-installed CUDA libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) to the DLL search path automatically.

## Usage

```powershell
python -m speech_coach.cli    # run a practice session
python -m pytest              # run the test suite
```

## Project structure

```
speech-coach/
├── speech_coach/        # the application package
├── tests/               # unit tests — no GPU needed, run in under a second
├── experiments/         # the experiments behind each design decision
├── data/                # recordings (git-ignored, never committed)
├── requirements.txt
└── pyproject.toml       # Ruff lint configuration
```

## Known limitations

- **Run-to-run variation:** filler counts can differ by about ±1 between transcriptions of the same audio. Compare trends over weeks, not single takes.
- **Approximate timestamps:** filler timing can't be measured precisely, so fillers are counted, not timed.
- **Vocabulary table:** corrections only fix mishearings that have already been seen and added. Real words that get misheard (e.g. "inspiring" for "aspiring") are left for the Phase 2 AI coach, which can use context.
- **Platform:** tested only on Windows with an NVIDIA GPU.

## Roadmap

- [x] **Phase 1 — Core loop:** record, transcribe, measure, scorecard
- [ ] **Phase 2 — AI coach:** a local LLM (via Ollama) gives feedback on structure and clarity, plus practice modes: behavioural interviews (STAR), impromptu speaking, "explain it simply", and retelling
- [ ] **Phase 3 — Web app:** browser UI usable from a phone, practice history, and progress charts
- [ ] **Phase 4 — Habit system:** daily practice prompts as phone notifications, and streaks
- [ ] **Phase 5 — Conversation mode:** a spoken AI conversation partner with follow-up questions
- [ ] **Phase 6 — Languages:** multilingual practice with per-language fillers and read-aloud pronunciation drills

## Credits and licence

- [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper) by Nyra Health — the speech model. **Licensed CC BY-NC 4.0 (non-commercial use only)**, which also limits how this project as a whole can be used.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — model inference, including Silero voice activity detection.
- This project's own code is released under the MIT Licence (see [`LICENSE`](LICENSE)).
