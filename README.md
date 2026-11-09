# Speech Coach
 
A local, privacy-first speaking practice app. Answer a question out loud, and Speech Coach measures **how** you said it (filler words, pauses, pace, repetitions, length), then a local AI coach gives feedback on **what** you said: your main point, your structure, and how you ended.
 
Everything runs on your own machine. Your voice never leaves it.
 
## Why I built this
 
Being an introvert, I wanted a private way to practise speaking (interview answers, explaining
technical ideas, everyday conversation) and get honest, measurable feedback instead of guessing
how I sounded. Eventually, I want to communicate naturally and effectively. Building it myself
was also a way to learn Python, local AI models and test-driven design. My end goal is a fully
interactive avatar I can have conversations with, one that evaluates how I communicate so I can
keep improving the way I speak.

## What a practice session looks like
 
```
Enter = 'Tell me about yourself.' | r = random | r <category> (behavioural, conversation, explain, opener, project) | or type your own question
> r explain
 
Question (explain): What is an access review, and why do companies do them?
Target length: 45–60 s
 
=== Transcript ===
An access review is [uh] a regular check where managers confirm who still needs access (pause 1.2s)
to each system and that keeps access {access} limited to what people need for their job ...
Key: [uh] filler   {and} repeated word   (pause 2.4s) pause of 1 s or more
 
=== Scorecard ===
Duration:         52.0 s (within the 45–60 s target)
Pace:             80 words/min (69 words)
Time pausing:     17.4 s of 52.0 s (33%)
Speaking rate:    120 words/min (pauses removed)
Fillers:          3 (3.5/min) {'uh': 3}
Possible fillers: {'basically': 2}
Repetitions:      access
Pauses >= 0.5 s:  17 (long >= 2 s: 0)
 
=== Coach ===
Main point:   An access review checks who has access and removes what isn't needed, for compliance and lower risk.
What worked:  "keeps access limited to what people need for their job"
              It links the definition straight to least privilege.
Fix next:     Name who performs the review before explaining why it matters.
You said:     "and that keeps access limited to what people need for their job"
Try:          "Managers and system owners do the review, and it keeps access limited to what each job needs."
Ending:       trailed off. Close on one summary sentence, then stop.
Next attempt: Definition first, then who does it, then two reasons, in under 60 seconds.
 
Enter = same question again | n = new question | q = quit:
```
*(Example output.)* Answer the same question again and you get a before/after table:
 
```
=== Compared with your last attempt ===
                    Before       Now
Fillers/min           9.3       3.5  better
Long pauses             1         0  better
Repetitions             0         1  worse
Duration             32 s      52 s
Speaking rate     132 wpm   120 wpm
```
 
## How it works
 
```
microphone ─► recorder ─► transcriber ─► cleanup ─► metrics ─► scorecard ─► AI coach ─► checks ─► feedback
 (Enter to     (16 kHz)   (CrisperWhisper  (silence   (fillers,                (Ollama,    (quotes,
  start/stop)              on the GPU)      check,     pace,                    local)      length,
                  │                         loop       repetitions)                          garbled)
                  │                         filter)        ▲                                    │
                  └──► pause detector (audio loudness) ────┘                                    ▼
                                                                              history (data/history.jsonl)
```
 
| Module | Responsibility |
|---|---|
| `recorder.py` | Records until you press Enter; trims the keypress click |
| `transcriber.py` | Runs CrisperWhisper via faster-whisper; the only file that knows about the speech model |
| `words.py` | The `Word` type, token cleanup, silence check and repetition-loop filter |
| `pauses.py` | Pauses and the speech span, measured from audio loudness |
| `metrics.py` | Deterministic speech metrics |
| `annotate.py` | Marks fillers, repeated words and pauses inside the transcript |
| `corrections.py` | Fixes known mishearings of my vocabulary in the displayed text |
| `pipeline.py` | Turns a recording into an analysis; shared by the CLI and experiments |
| `coach.py` | Prompt, response schema, deterministic checks on the coach's output, and the Ollama call |
| `questions.py` + `questions.toml` | Question bank by category, with target lengths |
| `compare.py` | Before/after comparison of two attempts |
| `history.py` | One JSON line per session, versioned |
| `cli.py` | The practice loop; connects the modules and displays results, with no logic of its own |
 
## Design decisions
 
Each decision below was tested, not assumed. The experiments behind them are in [`experiments/`](experiments/).
 
### Measuring speech
 
1. **CrisperWhisper instead of standard Whisper.** Standard Whisper is trained to write what you *meant*, so it silently drops "um", "uh" and stutters, which are exactly what this app measures. The well-known prompt trick didn't bring them back; CrisperWhisper, a verbatim fine-tune of Whisper, kept them. ([`filler_test.py`](experiments/filler_test.py))
2. **No AI in the scoring.** Metrics are plain, deterministic Python. The same transcript always gives the same numbers, so progress can be tracked and every rule is unit-tested.
3. **Certain vs. possible fillers.** "Um" is always a filler; "like" and "basically" often aren't. Only certain fillers count toward the score.
4. **Time comes from the audio, not the word timestamps.** Word timestamps from faster-whisper run back to back, with silence absorbed into the words. Pauses and the answer's duration are both measured from loudness in 30 ms frames, with a threshold relative to each recording's own speaking level. Using one source for both guarantees that pause time can never exceed the duration, and a test checks that invariant.
5. **Two defences against hallucinated words, cross-checked against the audio.**
   - *Invented words at the end:* any word that starts without at least 0.1 s of sound after it is dropped.
   - *Repetition loops:* disabling conditioning on previous text stopped a 60-word "A S A S…" loop ([`loop_test.py`](experiments/loop_test.py)), and a filter drops any 1–3 word pattern repeated 5+ times as a safety net.
   Every removed word is printed, so nothing disappears silently.
6. **Vocabulary fixes apply to the display only.** Giving the model a vocabulary hint backfired: `hotwords` dropped two-thirds of a recording, and `initial_prompt` cost 20% of detected fillers ([`vocab_test.py`](experiments/vocab_test.py)). A lookup table corrects known mishearings ("Bemo" → "BMO", "a m l" → "AML") in the text you read, while metrics use the original words.
### The AI coach: it advises, the code decides
 
7. **The coach interprets; it never measures.** It receives the transcript and the scorecard numbers and judges what software can't: the main point, structure, clarity and the ending. It is told never to count or calculate anything itself.
8. **Every claim the coach makes about your words is checked.** The response is forced into a JSON schema (pydantic, enforced by Ollama), then verified in code: quotes must really appear in the transcript, must be under 30 words, must not quote garbled recognition, and the praised sentence can't also be the one it rewrites. If a check fails, the coach gets one retry with the specific problems; anything still unverified is hidden, never shown.
9. **The coach only runs when there's something to coach.** Below 15 words, the app skips the coach instead of letting it invent advice for an answer that wasn't given. If the coach fails for any reason (empty response, invalid JSON, Ollama not running), the error is reported and the session is still scored and saved.
10. **Your transcript is data, not instructions.** It's wrapped in tags, and angle brackets are stripped so it can't close its own tag and pose as instructions. A test writes out that attack and proves it fails.
11. **The schema shapes the reasoning.** The model fills fields in order, so `main_point` comes first: it states what you said before it judges it. `ended_clearly` is a boolean so the history can track it over time.
12. **The model is chosen by measurement, not by release date.** With thinking on, qwen3.5:9b used over 4,000 tokens on a technical answer and ran out of room; qwen3:14b finished in 416 tokens and 18.6 s on the same recording ([`think_debug.py`](experiments/think_debug.py)). Thinking is off by default (about 5 s per review) and available with `--think`.
### Data
 
13. **Local-first.** Recordings and history live in `data/`, which is excluded from Git.
14. **Versioned history.** Each session is one line of JSON in `data/history.jsonl`, with a `schema_version` and the coach's `prompt_version`, so sessions recorded under different rules can always be told apart.
## Setup
 
**Tested on:** Windows 11, Python 3.12, NVIDIA RTX 5060 Ti (16 GB).
 
**1. Install [Ollama](https://ollama.com) and pull the coach model:**
 
```powershell
ollama pull qwen3:14b
```
 
**2. Set up the project:**
 
```powershell
git clone https://github.com/calwong88/speech-coach.git
cd speech-coach
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
 
The first run downloads the CrisperWhisper model (a few GB) from Hugging Face.
 
**Notes:**
- **GPU memory:** the speech model and the coach together use about 15 GB. Check with `ollama ps` that the coach runs at `100% GPU`; if it doesn't, lower `NUM_CTX` in `coach.py`.
- **RTX 50-series (Blackwell):** the speech model runs in `float16`; `int8` can crash on this architecture. On Windows, `transcriber.py` adds the pip-installed CUDA libraries to the DLL search path automatically.
- **No audio recorded?** Check that the right microphone is the Windows default input, that desktop apps are allowed microphone access (Settings → Privacy & security → Microphone), and that no other app is holding the microphone. `python -m sounddevice` lists devices; the default input is marked `>`.
## Usage
 
```powershell
python -m speech_coach.cli            # practise (coach thinking off: about 5 s per review)
python -m speech_coach.cli --think    # slower, sometimes more concrete coaching
python -m pytest                      # run the tests
```
 
At the question prompt: **Enter** for "Tell me about yourself", **`r`** for a random question, **`r explain`** (or `opener`, `behavioural`, `project`, `conversation`) for a random question from one category, or type your own. After each answer: **Enter** to try the same question again, **`n`** for a new question, **`q`** to quit.
 
To add questions, edit [`speech_coach/questions.toml`](speech_coach/questions.toml). A test validates the file, so a typo shows up in `pytest`, not mid-practice.
 
## Tests
 
88 tests run in under a second, with no GPU or model needed. They include:
- **Regression fixtures from real recordings:** every odd token, invented ending and repetition loop found in practice became a test case.
- **An invariant:** total pause time can never exceed the answer's duration.
- **A security test:** a transcript trying to escape its tag.
- **A data test:** the question bank file is validated like code.
The AI coach itself is evaluated with experiments in `experiments/`, not unit tests, because a language model's output isn't deterministic enough to assert on. Everything deterministic around it is unit-tested.
 
## Project structure
 
```
speech-coach/
├── speech_coach/        # the application package (including questions.toml)
├── tests/               # unit tests: no GPU, under a second
├── experiments/         # the experiments behind each design decision
├── data/                # recordings and history (git-ignored, never committed)
├── requirements.txt
└── pyproject.toml       # Ruff and pytest configuration
```
 
## Known limitations
 
- **Run-to-run variation:** filler counts can differ by about ±1 between transcriptions of the same audio. Compare trends across sessions, not single takes.
- **Approximate positions:** word timestamps are approximate, so pause markers in the annotated transcript show roughly where a pause happened.
- **Vocabulary table:** corrections only fix mishearings that have already been seen and added.
- **The local coach isn't perfect.** It sometimes contradicts itself (praising and criticizing the same sentence) or adds wording you didn't say to a rewrite. The checks catch what can be checked; for the rest, only use a rewrite if it's true for you.
- **`--think`** is slower, and on long reasoning it can still run out of room. When that happens, the session is still scored and saved.
- **Platform:** tested only on Windows with an NVIDIA GPU.
## Roadmap
 
- [x] **Phase 1, core loop:** record, transcribe, measure, scorecard
- [x] **Phase 2, AI coach:** verified feedback with main point and ending, practice loop with attempt comparison, annotated transcript, question bank with target lengths, versioned history
- [ ] **Coach model comparison:** local models vs. a hosted API model on the same saved sessions
- [ ] **Phase 3, web app:** browser UI usable from a phone, progress charts from the history
- [ ] **Phase 4, habit system:** daily practice prompts as phone notifications, streaks
- [ ] **Phase 5 — Conversation mode:** a spoken AI conversation partner with follow-up questions, working toward an interactive avatar
- [ ] **Phase 6, languages:** multilingual practice with per-language fillers and read-aloud drills
## Credits and licence
 
- [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper) by Nyra Health: the speech model. **Licensed CC BY-NC 4.0 (non-commercial use only)**, which also limits how this project as a whole can be used.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): speech model inference, including Silero voice activity detection.
- [Ollama](https://ollama.com) and [Qwen3](https://github.com/QwenLM/Qwen3): the local coach.
- [pydantic](https://docs.pydantic.dev): validation of the coach's responses.
- This project's own code is released under the MIT Licence (see [`LICENSE`](LICENSE)).
 