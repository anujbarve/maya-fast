# Maya2 Marathi Compression Research (Side Plan) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out how far Maya2 can be shrunk for Marathi-only TTS on a 16 GB Apple Silicon Mac, and how much speech quality each step costs, with measured numbers only.

**Architecture:** Research track, parallel to and separate from `2026-10-27-maya2-cpu-tts.md`. It never runs a model while the main plan's spike, sweep or benchmark is running (one model process at a time, 16 GB RAM). It reuses the main plan's round-trip scorer and its Marathi sentences. Order: audit (read-only, no extra model runs) → INT8 comparison → INT4 feasibility → vocabulary pruning → distillation scoping (document only). Each step has a go/stop gate; a failed gate stops the track and reports.

**Tech Stack:** Python 3.10-3.12, PyTorch CPU (from `.venv-tts`), `safetensors`, `transformers` (pinned by upstream), IndicConformer ASR via `src/models.py`, `src/metrics.py`, `src/audio_utils.py`. INT4 feasibility may need GGUF (llama.cpp) or MLX tooling, decided in Task S3.

**Spec:** No spec file. Source: user brief "Build a Lightweight, Marathi-First Version of Maya 2 Native TTS" (2026-10-27, pasted into the conversation). Its sections 3-9 are reduced here to what can be measured on this Mac; the rest is listed under Out of Scope.

## Global Constraints

- Primary objective (from brief): maximize Marathi speech quality per gigabyte and per unit of inference compute.
- Target: 16 GB Apple Silicon Mac, CPU/unified memory. Peak RAM under 12 GB per model process. Never run two model processes at once.
- Scoring: round-trip CER/WER through IndicConformer are necessary but not sufficient (brief §8). Every config also gets a human listening score on a fixed sentence set, reported separately. CER/WER alone never decides a gate.
- No benchmark number is written into a report unless it came from a run on the config it describes. No projected numbers, ever.
- Upstream Maya2 source is read-only at `scratchpad/m2`. Converted or pruned weights live outside the repo (Hugging Face cache or a scratchpad dir) and are never committed.
- Test set: the main plan's Marathi sentences plus conversational, long, proper-noun, number and code-switch sentences added in S0. Fixed, versioned in `benchmarks/tts_testset_mr.json`.
- Commit trailer: `Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Audit counts shared weights as language-specific:** a component is "Marathi-only removable" only if the audit shows no use outside Marathi tokens. Test: `test_audit_flags_shared_tensors` (Task S1).
2. **Pruned vocabulary breaks a Marathi word:** any token needed by the test set must survive pruning, and round-trip must show it. Test: `test_pruned_vocab_covers_testset` (Task S4).
3. **INT4 reported as speedup without a kernel:** INT4 storage must be paired with a measured runtime on the same hardware, or reported as storage only. Test: `test_int4_report_has_runtime_or_storage_only` (Task S3).
4. **Human score missing or unblinded:** listening results must include rater count and blind A/B labels. Protocol check in Task S5.

---

### Task S0: Test set

**Files:**
- Create: `benchmarks/tts_testset_mr.json`
- Test: none (data file, validated by the scorer in S2)

**Interfaces:**
- Produces: JSON list of `{"id": str, "category": "conversational"|"long"|"proper_noun"|"number"|"code_switch", "text": str}`. At least 5 per category.

- [ ] **Step 1: Write the file** with the categories above. Include the main plan's benchmark sentence in `conversational`.
- [ ] **Step 2: Validate** it parses and has at least 5 entries per category: `python -c "import json,collections;d=json.load(open('benchmarks/tts_testset_mr.json'));print(collections.Counter(x['category'] for x in d))"`. Expected: five or more per category.
- [ ] **Step 3: Commit** `git add benchmarks/tts_testset_mr.json && git commit -m "data(tts): add fixed Marathi TTS test set"`.

---

### Task S1: Architecture audit (read-only)

**Files:**
- Create: `benchmarks/tts_audit.py`
- Create: `docs/superpowers/plans/2026-10-27-maya2-audit-results.md` (results, filled in by running the script)

**Interfaces:**
- Produces: `audit(ckpt_dir: str) -> dict` returning `{"components": {name: {"params": int, "bytes_bf16": int}}, "vocab": {"size": int, "embed_params": int, "lm_head_params": int}, "total_params": int}`. Components: `backbone`, `depth`, `mimi`, `ecapa`, `embed_tokens`, `lm_head`, `cb0_head`, `other`. Read tensor headers from safetensors only; never instantiate the model.
- Consumes: safetensors files from `maya3.hub.resolve("maya-research/maya2-native")`.

- [ ] **Step 1: Write failing test** `tests/test_tts_audit.py::test_audit_flags_shared_tensors`: build a tiny fake safetensors file in a temp dir with a tensor named `model.embed_tokens.weight` and one `depth.groups.0.weight`; assert `audit()` reports both, and that `embed_tokens` is marked `shared_with_audio=False`. Run: `.venv/bin/python -m unittest tests/test_tts_audit.py` → expected FAIL.
- [ ] **Step 2: Implement `audit()`** reading `safetensors.safe_open(path, "pt")` keys and shapes without loading tensor data. Group by the component prefixes above.
- [ ] **Step 3: Run the test** → PASS.
- [ ] **Step 4: Run on the real checkpoint** (after the main plan's Task 1 download is complete and its run has finished): `.venv-tts/bin/python benchmarks/tts_audit.py`. Write the output table into `docs/superpowers/plans/2026-10-27-maya2-audit-results.md`.
- [ ] **Step 5: Gate.** Go if the audit shows the `embed_tokens` + `lm_head` share of total params. Stop and report if the audit shows no separable language-specific component: then S4 is dropped.
- [ ] **Step 6: Commit** the script, test and results file.

---

### Task S2: INT8 comparison (uses the main plan's sweep results)

**Files:**
- Create: `benchmarks/tts_int8_compare.py`
- Modify: `docs/superpowers/plans/2026-10-27-maya2-cpu-tts.md` (add a `## Compression results` section pointer; do not change its tasks)

**Interfaces:**
- Consumes: `synthesize_once` from Task 2 of the main plan, `score_transcript` from `src/metrics.py`.
- Produces: one row per config `{name, size_mb, peak_rss_mb, rtf, first_audio_s, cer, wer, human_score}` for: `bf16-baseline`, `int8-backbone`, `int8-backbone+depth`.

- [ ] **Step 1: Run each config** on `benchmarks/tts_testset_mr.json`, one process per config. Size measured as on-disk bytes of the saved weights, not theoretical.
- [ ] **Step 2: Record** rows in the results file. `human_score` stays blank until S5.
- [ ] **Step 3: Gate.** Go to S3 only if `int8-backbone` keeps CER within 0.02 of baseline and cuts on-disk size by at least 40%. Otherwise stop and report.
- [ ] **Step 4: Commit.**

---

### Task S3: INT4 feasibility (spike, runtime decided here)

**Files:**
- Create: `benchmarks/tts_int4_spike.md` (decision record: runtime chosen, why, what it cost)

**Interfaces:**
- Produces: a decision among (a) torch INT4 CPU kernel, (b) GGUF via llama.cpp, (c) MLX port. Each option's blocker is written down before choosing.

- [ ] **Step 1: Check each option** against the upstream model code: which ops the backbone (Qwen3-1.7B shape), depth decoder and Mimi need. Record unsupported ops per option.
- [ ] **Step 2: Choose the smallest option with no unsupported ops.** If all options need reimplementing the depth loop, record INT4 as "storage-only, no runtime" and stop the track here. Do not claim a speedup.
- [ ] **Step 3: If an option is chosen**, quantize and run S2's test set. Report size, RSS, RTF and CER. Storage-only results are labelled as such.
- [ ] **Step 4: Commit the decision record.**

---

### Task S4: Vocabulary pruning (only if S1 passes)

**Files:**
- Create: `benchmarks/tts_vocab_prune.py`
- Create: `tests/test_tts_vocab_prune.py`

**Interfaces:**
- Produces: `prune_vocab(token_ids_needed: set[int], embed: "Tensor", lm_head: "Tensor") -> tuple[Tensor, Tensor, dict[int,int]]` returning the reduced matrices and an old→new id map. Consumes the tokenizer and the token ids the test set actually produces (computed by tokenizing `tts_testset_mr.json`).

- [ ] **Step 1: Write failing test** `test_pruned_vocab_covers_testset`: given a tiny embed matrix and a set of needed ids, assert every needed id appears in the map and the reduced shape is `(len(needed), dim)`. Expected FAIL.
- [ ] **Step 2: Implement `prune_vocab`.** Keep special and control tokens used by the serving path. Those ids come from the model config, not the test set.
- [ ] **Step 3: Run the test** → PASS. Then run round-trip on the test set with the pruned model. Gate: CER within 0.02 of unpruned and no test sentence tokenizes to an unmapped id.
- [ ] **Step 4: Commit.**

---

### Task S5: Human listening protocol

**Files:**
- Create: `docs/superpowers/plans/2026-10-27-maya2-listening-protocol.md`

**Interfaces:**
- Produces: a blind A/B protocol: fixed pairs of config outputs for the same sentence, randomized left/right, with rater ID and 1-5 rating per axis (naturalness, pronunciation, voice consistency). Minimum 3 raters, 10 sentences per config pair.

- [ ] **Step 1: Write the protocol** and the generator that writes the shuffled pairs with a hidden key file outside the repo.
- [ ] **Step 2: The user runs the listening tests.** This step cannot be done by the agent. The task stays open until results are pasted back.
- [ ] **Step 3: Commit** the protocol and generator.

---

## Out of Scope (this plan)

- Full distillation into a smaller Marathi architecture. Separate research track, scoped in a later plan if S2 and S4 leave a gap.
- Making Maya2 run on the Intel i5 target. Not feasible per the main plan's gate.
- Fine-tuning on Marathi data. Needs data and compute not in this repo.

## Self-review

- **Spec coverage:** audit (S1), INT8 (S2), INT4 (S3), language specialization (S4), human and voice-consistency evaluation (S5), test set (S0). Distillation and fine-tuning are listed as out of scope, not silently dropped.
- **Type consistency:** `audit`, `prune_vocab`, `synthesize_once` and `score_transcript` named the same wherever used.
- **Gates:** every task after S1 has an explicit stop condition, so a failed branch ends the side track instead of drifting.
