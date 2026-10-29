# Maya2 CPU Marathi TTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speak Marathi text from this repo using `maya-research/maya2-native`, running on the CPU of an Apple M-series Mac, as fast and as light as measurement allows.

**Architecture:** Maya2's shipped serving engine (`maya3.engine.Engine`) is CUDA-only, so bypass it. The upstream repo also contains a plain-PyTorch reference decoder (`scripts/s2_serve_controls.py: ServeSession`, `depth_generate`, `prompt_from_example`) that already has a non-CUDA branch (fp32 when the device is not `cuda`). Build a small single-request synthesizer on that reference decoder, measure it, then apply optimizations in order of measured gain. A go/no-go gate after the spike decides whether the rest of the plan runs.

**Tech Stack:** Python 3.10-3.12, PyTorch 2.8.0 (CPU/arm64 wheel), `transformers==5.6.2`, `speechbrain==1.1.2`, upstream `maya3` package (installed from GitHub, pinned commit), `soundfile`, existing repo ASR (`IndicConformerONNXASR`) for round-trip scoring.

**Spec:** No spec file. Requirements come from the 2026-10-27 conversation: use `maya-research/maya2-native` for Marathi TTS; run on this Mac (Apple M5, 16 GB) on CPU only; make it faster and less demanding. Upstream facts used here were read from a clone of `https://github.com/MayaResearch/maya2-native` (commit to be pinned in Task 1).

## Global Constraints

- **Hardware decision (my call, flag if wrong):** the TTS targets Apple M-series, CPU only (`device="cpu"`; no MPS, no CUDA). It is **not** supported on the Intel i5 8th-gen U target in `memory/target-hardware.md`; nothing here may be claimed to work there. ASR stays unaffected: TTS code is optional and must not be imported by `src/cli.py`, `src/listen.py`, `src/models.py`, or `benchmarks/run_benchmark.py`.
- Peak RAM must stay under 12 GB on the 16 GB Mac (leave room for the OS and the ASR model, about 0.6 GB).
- Maya2 output is 24 kHz mono float32. Do not resample inside the synthesizer.
- TTS dependencies go in a new `requirements-tts.txt`, never in `requirements.txt` (torch 2.8.0 and transformers 5.6.2 are heavy and pinned).
- The weights are about 5 GB (BF16 safetensors): download only on explicit first use, never in tests, and never inside the repo (use the Hugging Face cache).
- Tests use `unittest`, run with `.venv/bin/python -m unittest`, and must not download models, import torch, or write outside a temp dir. Fake the backend.
- Maya2 weights are Apache-2.0; Mimi is CC-BY-4.0. README must say so. The model adds no audio watermark; README must say that too.
- Every commit message ends with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Empty or whitespace-only text:** must raise `ValueError`, not hang the generation loop. Test: `test_empty_text_rejected` (Task 3).
2. **Model never emits the stop token (runaway generation):** output must be bounded by a frame cap derived from text length, not run for 120 s. Test: `test_frame_cap_scales_with_text_length` (Task 3).
3. **Text with digits, ASCII, or mixed Devanagari/Latin:** goes through upstream `maya3.text.normalise_text(text, "mr")` before synthesis. Test: `test_text_is_normalized_before_synthesis` (Task 3).
4. **Long paragraph:** split into pieces with the upstream splitter and joined with a short silence, not one giant generation. Test: `test_long_text_is_split_and_joined` (Task 3).
5. **Missing optional dependencies (`torch`, `maya3`):** CLI prints one actionable line (`pip install -r requirements-tts.txt`) and exits 1, no traceback. Test: `test_missing_dependency_message` (Task 4).

---

### Task 1: Spike and go/no-go gate (throwaway)

**Files:**
- Create: `requirements-tts.txt`
- Create: `benchmarks/tts_spike.py` (throwaway; deleted in Task 3)

**Interfaces:**
- Produces: a measured baseline (load seconds, peak RSS MB, audio seconds, wall seconds, RTF, per-phase seconds for backbone / depth / Mimi) and a short written verdict appended to the bottom of this plan under `## Spike result`.

- [ ] **Step 1: Create the venv and pin upstream**

Create `.venv-tts` (separate from `.venv`). Write `requirements-tts.txt` with: `torch==2.8.0`, `torchaudio==2.8.0`, `transformers==5.6.2`, `speechbrain==1.1.2`, `tokenizers>=0.22,<0.23`, `scipy>=1.13`, `safetensors>=0.6`, `pyyaml>=6.0`, `huggingface_hub>=1.0`, and `maya3 @ git+https://github.com/MayaResearch/maya2-native@<COMMIT_SHA>` where the SHA is the current `main` head, recorded in the file. Install with `.venv-tts/bin/pip install -r requirements-tts.txt` (arm64 macOS wheels, default PyPI index, not the cu128 index).
Verify: `.venv-tts/bin/python -c "import maya3, torch; print(torch.__version__)"` prints `2.8.0`.

- [ ] **Step 2: Write `benchmarks/tts_spike.py`**

CLI: `python benchmarks/tts_spike.py "<marathi text>" --q-serve 16 --dtype fp32 --threads 8 -o out.wav`. It does what `Engine.__init__`, `prepare_voice` and `_plan` do, minus CUDA:
1. `maya3.hub.resolve("maya-research/maya2-native")` for `ckpt`, `backbone_dir`, `mimi_dir`, `voices_json`.
2. `tts.model.load_s3.load_model_for_eval(ckpt, device="cpu", dtype=<fp32|bf16>, cfg={"model": {"backbone": backbone_dir, "attn": "sdpa"}, "data": {"max_tokens": 8192}}, config=<upstream>/configs/s3_fresh_v1.yaml, amortize_frac=1.0, require_anchor=True, verbose=False)`.
3. `PackerV2(model.backbone.tokenizer, model.backbone.token_ids, max_tokens=8192)`, `MimiModel.from_pretrained(mimi_dir).eval()`.
4. Load named voice `Charvi` via `maya3.prompt.load_named_voices` into a `VoiceRegistry`; build its prefix and speaker memory exactly as `Engine.prepare_voice` does (`builder.prefix`, `model.depth.spk_memory`), without the CUDA-side KV caching.
5. Build the prompt with `scripts.s2_serve_controls.prompt_from_example(builder.example(packer, voice, text, "mr"), model, "cpu", dtype, q_serve)`.
6. Run `ServeSession(model, voice.mem, voice.rmask, q_serve=q, device="cpu").speak(prompt, cap, stop="controlled", min_frames=...)` with `cap` and `min_frames` computed as in `Engine._plan` (`expected_frames`, prior from `configs/text_frames_prior.json`).
7. Decode with `mimi.decode(codes[None])`, write 24 kHz WAV, print the metrics. Wrap the three phases in `time.perf_counter()` timers by monkeypatching `ServeSession._step`, `depth_generate` and `mimi.decode`.

Where an upstream call hard-requires CUDA, patch it in the spike and record every patch under `## Spike result`; that list is the true port cost.

- [ ] **Step 3: Run on the benchmark sentence**

Run with text `पोलीस अधीक्षक चंद्रशेखर सोळंकी यांनी सांगितले की आरोपी चेहरा झाकून घेऊन कोर्टात हजर झाला` (the repo's own clean sample), once with `--dtype fp32` and once with `--dtype bf16`, `--q-serve 16`.
Expected: a WAV that is audible Marathi speech and a metrics line per run.

- [ ] **Step 4: Round-trip check**

Transcribe the WAV with `IndicConformerONNXASR` (resample 24k to 16k with `src.audio_utils.resample`) and score against the input text with `src.metrics.score_transcript`. Record CER.

- [ ] **Step 5: Gate and commit**

Append `## Spike result` (numbers, patches list, verdict) to this plan. **Go** if fp32 or bf16 RTF is at most 3.0 and peak RSS is under 12 GB and round-trip CER is at most 0.15 (these thresholds are my guess; adjust them with the user). **No-go** otherwise: stop, report numbers, and recommend the light ONNX TTS instead. Do not start Task 2 without a Go.

```bash
git add requirements-tts.txt benchmarks/tts_spike.py docs/superpowers/plans/2026-10-27-maya2-cpu-tts.md
git commit -m "spike(tts): measure Maya2 on M-series CPU"
```

---

### Task 2: Measured optimization sweep

**Files:**
- Create: `benchmarks/tts_benchmark.py`
- Modify: `docs/superpowers/plans/2026-10-27-maya2-cpu-tts.md` (results table)

**Interfaces:**
- Consumes: the spike's synthesis function, moved into `benchmarks/tts_benchmark.py` as `synthesize_once(text: str, *, q_serve: int, dtype: str, threads: int, quant: str) -> tuple[np.ndarray, dict]` returning the 24 kHz waveform and a metrics dict with keys `load_s`, `wall_s`, `audio_s`, `rtf`, `peak_rss_mb`, `backbone_s`, `depth_s`, `mimi_s`.
- Produces: a chosen default config `(q_serve, dtype, threads, quant)` written into the results table; Task 3 reads these defaults.

- [ ] **Step 1: Write `benchmarks/tts_benchmark.py`**

Runs `synthesize_once` over a fixed list of 3 Marathi sentences (short, medium, long, defined in the file) for each config in a grid, then scores each output by ASR round-trip (`score_transcript`), and writes `benchmarks/tts_results.csv` (columns: config, the metric keys, `cer`, `wer`). Order of the sweep, one variable at a time from the Task 1 baseline, keeping a change only if RTF drops at least 10% and mean round-trip CER rises by at most 0.02 absolute:
  1. `threads` in {4, 8, 10} (`torch.set_num_threads`).
  2. `dtype` fp32 vs bf16 for backbone and depth separately.
  3. `q_serve` in {16, 12, 8}. Fewer codebooks is the largest lever; Mimi decodes only the served codebooks.
  4. `quant`: `none` vs `torch.ao.quantization.quantize_dynamic(model.backbone, {torch.nn.Linear}, dtype=torch.qint8)` (backbone only; the depth decoder is checked separately and kept in float if CER regresses).
  5. KV/graph overhead: `torch.inference_mode` is already on in `ServeSession.speak`; try `torch.compile(model.backbone)` only if steps 1-4 leave RTF above 1.0, and keep it only if it also survives a second fresh-process run.

- [ ] **Step 2: Run the sweep on this Mac**

Run: `.venv-tts/bin/python benchmarks/tts_benchmark.py`
Expected: `benchmarks/tts_results.csv` with one row per config; the script prints the winning config.

- [ ] **Step 3: Record and commit**

Paste the results table and the chosen default config into this plan under `## Sweep result`, including every config that was rejected and why.

```bash
git add benchmarks/tts_benchmark.py benchmarks/tts_results.csv docs/superpowers/plans/2026-10-27-maya2-cpu-tts.md
git commit -m "bench(tts): sweep threads, dtype, codebooks and int8 on M-series CPU"
```

---

### Task 3: `MayaCpuTTS` module

**Files:**
- Create: `src/tts_models.py`
- Create: `tests/test_tts_models.py`
- Delete: `benchmarks/tts_spike.py`

**Interfaces:**
- Produces, in `src/tts_models.py`:
  - `class BaseMarathiTTS(ABC)` with `load(self) -> None`, `synthesize(self, text: str) -> np.ndarray` (float32 mono), `sample_rate` property `-> int`, `model_id` property `-> str`.
  - `class MayaCpuTTS(BaseMarathiTTS)` with `__init__(self, voice: str = "Charvi", q_serve: int = <Task 2>, dtype: str = <Task 2>, threads: int = <Task 2>, quant: str = <Task 2>, backend=None)`. `backend` is an injection point for tests: an object with `generate_piece(text: str, max_frames: int) -> np.ndarray` and `normalize(text: str) -> str` and `split(text: str) -> list[str]` and `expected_frames(text: str) -> int`. When `None`, `load()` builds the real backend from the Task 1 code path (imports `torch` and `maya3` lazily, inside `load()` only).
  - `get_tts(name: str) -> BaseMarathiTTS` for `"maya2-cpu"`; raises `ValueError` for other names, mirroring `get_model` in `src/models.py`.
  - Constants `PIECE_GAP_S = 0.15` and `SAMPLE_RATE = 24000`.
- Frame cap rule: `max_frames = max(8, int(2.0 * expected_frames(piece)) + 40)` (same as upstream `Engine._plan` `loose`), capped at 1500.

- [ ] **Step 1: Write failing tests in `tests/test_tts_models.py`**

A `FakeBackend` returns `np.ones(frames_requested * 1920, np.float32)` (1920 samples is 80 ms at 24 kHz), records every `text` and `max_frames`, and uses `normalize=str.upper`-style marker (`lambda t: "N:" + t`), `split` that splits on `"।"`, and `expected_frames=lambda t: len(t)`. Tests:
  - `test_empty_text_rejected`: `synthesize("")` and `synthesize("   ")` raise `ValueError`.
  - `test_text_is_normalized_before_synthesis`: the fake's recorded text starts with `"N:"`.
  - `test_long_text_is_split_and_joined`: input with two `।`-separated pieces yields output length equal to the two pieces plus `int(PIECE_GAP_S * SAMPLE_RATE)` zeros between them (assert the gap samples are exactly `0.0`).
  - `test_frame_cap_scales_with_text_length`: recorded `max_frames` for a 10-char piece equals `max(8, 2*10+40)`, and for a 2000-char piece equals `1500`.
  - `test_get_tts_unknown_name`: `get_tts("nope")` raises `ValueError`.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m unittest tests/test_tts_models.py`
Expected: FAIL (`ModuleNotFoundError: src.tts_models`).

- [ ] **Step 3: Implement `src/tts_models.py`**

`synthesize`: strip, reject empty, `normalize`, `split`, per piece compute `max_frames`, call `generate_piece`, join with `PIECE_GAP_S` zeros. The real backend is the Task 1 spike code reorganized as a private class `_MayaCpuBackend` in the same file (one `load()` that does the model, packer, Mimi and voice setup once; `generate_piece` that builds the prompt, runs `ServeSession.speak`, decodes with Mimi, returns float32). Apply the Task 2 winning config.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m unittest tests/test_tts_models.py`
Expected: PASS, and `.venv/bin/python -m unittest discover tests` still passes (ASR suite unaffected, no torch import).

- [ ] **Step 5: Real-model smoke check, then commit**

Run once in `.venv-tts`: `.venv-tts/bin/python -c "from src.tts_models import get_tts; import soundfile as sf; t=get_tts('maya2-cpu'); t.load(); sf.write('/tmp/m.wav', t.synthesize('नमस्कार, तुम्ही कसे आहात?'), 24000)"`
Expected: a playable Marathi WAV. Then delete the spike file.

```bash
git rm benchmarks/tts_spike.py
git add src/tts_models.py tests/test_tts_models.py
git commit -m "feat(tts): add Maya2 CPU synthesizer behind a small interface"
```

---

### Task 4: `speak` CLI and docs

**Files:**
- Create: `src/speak.py`
- Modify: `tests/test_tts_models.py` (CLI test), `README.md` (new section 6 "Text to speech"), `.gitignore` (add `.venv-tts/`)

**Interfaces:**
- Consumes: `get_tts`, `BaseMarathiTTS` from Task 3.
- Produces: `src/speak.py` with `main(argv: list[str] | None = None) -> int`; CLI `python3 src/speak.py "<text>" -o out.wav [--play] [--model maya2-cpu]`. `--play` uses `sounddevice` (already a dependency). Exit code 1 and one stderr line `TTS dependencies missing: pip install -r requirements-tts.txt` when importing `torch` or `maya3` fails.

- [ ] **Step 1: Write failing test `test_missing_dependency_message`**

Patch `src.speak.get_tts` to raise `ImportError("torch")`; call `main(["नमस्कार", "-o", <tmp>])`; assert return value `1` and that captured stderr is exactly the line above and contains no `Traceback`.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m unittest tests/test_tts_models.py -k missing_dependency`
Expected: FAIL (`ModuleNotFoundError: src.speak`).

- [ ] **Step 3: Implement `src/speak.py`**

Follow `src/cli.py` conventions (argparse, `sys.path.insert` of the project root). Positional `text`; `-o/--out` default `out.wav`; `--play`; `--model` default `maya2-cpu`. Print the same one-line timing summary as `cli.py` style: audio seconds, wall seconds, RTF.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m unittest discover tests`
Expected: all PASS.

- [ ] **Step 5: README and commit**

README section 6 states: M-series CPU only, not supported on the i5 target, 5 GB first-use download, install via `requirements-tts.txt`, measured RTF and RAM from Task 2 (labelled as measured on this Mac), Apache-2.0 weights plus CC-BY-4.0 Mimi, and that the model adds no audio watermark.

```bash
git add src/speak.py tests/test_tts_models.py README.md .gitignore
git commit -m "feat(tts): add speak CLI and README section"
```

---

## Self-review

- **Spec coverage:** use Maya2 for Marathi TTS (Tasks 1, 3, 4); M-series CPU only (Global Constraints, Task 1 device `cpu`); faster and less demanding (Task 2 sweep, gate in Task 1, RAM limit in constraints). No gaps found.
- **Type consistency:** `get_tts`, `BaseMarathiTTS`, `MayaCpuTTS`, `synthesize_once` and `generate_piece` are named the same wherever they appear. Defaults in Task 3 come from the Task 2 table.
- **Known unknowns, called out instead of guessed:** whether `load_model_for_eval` and `prepare_voice`-equivalent code run on CPU without more patches than Task 1 lists; the real RTF; the pinned upstream SHA; the Go thresholds. Task 1 exists to answer these before anything permanent is written.
- **Proportion:** four tasks; only Task 3 and 4 add permanent code, and both stay small.

## Spike result

Machine: Apple M5, 16 GB, arm64, CPU only, 8 threads, Python 3.13, torch 2.8.0, upstream maya2-native `c6b376868e268551d065c4206193e96257bef4a1`. Sentence: the repo's clean sample. Voice Charvi, q_serve 16, one run each (seeded 0).

| dtype | load s | frames | audio s | wall s | RTF | peak RSS MB | backbone s | depth s | Mimi s | round-trip CER |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 94.3 (includes ECAPA/Mimi fetch) | 85 | 6.80 | 48.8 | 7.18 | 7981 | 22.7 | 25.1 | 0.7 | 0.000 |
| bf16 (backbone bf16, depth fp32 as upstream Engine) | 23.7 | 84 | 6.72 | 31.0 | 4.62 | 7480 | 8.5 | 21.9 | 0.5 | 0.000 |

Round-trip: IndicConformerONNXASR (24k -> 16k via `src.audio_utils.resample`) returned the input text verbatim for both; WER 0.0, CER 0.0.

Upstream patches applied: **none required for CUDA**. `load_model_for_eval`, `prompt_from_example`, `ServeSession`, `depth_generate` and Mimi decode all run on `device="cpu"`. Only spike-side deviations from `Engine`: (1) skipped CUDA-only fast paths (StaticBackbone, FastDepth, StaticMimi, CUDA graphs, prefix-KV cache), so this is the slow eager reference path; (2) added `model.depth.float()` after load, exactly as `Engine.__init__` does (without it bf16 fails with a float/bf16 dtype mismatch in `depth.spk_memory`). Port cost beyond that is a CPU re-implementation of the fast path.

Time split: depth decoding (31 sequential levels per frame, eager, fp32) is the bottleneck, 22-25 s of 31-49 s. Backbone drops 22.7 -> 8.5 s in bf16. Remaining levers (untested): fewer depth levels via lower q_serve, bf16/int8 depth, torch.compile, thread count. Download note: the 5 GB shard stalled twice on the xet backend and a transient DNS failure; `HF_HUB_DISABLE_XET=1` completed it.

**Verdict: NO-GO** under the written gate (RTF <= 3.0). Best RTF is 4.62 (bf16), a miss of 1.5x. RSS (7.5-8.0 GB < 12 GB) and CER (0.0 <= 0.15) pass. Controller decision: either accept a looser RTF threshold (the 3.0 was a guess) or try the cheap levers above in a follow-up spike, or fall back to the light ONNX TTS. Tasks 2+ not started.
