# maya-fast

Research notes and throwaway spike scripts for making the Maya2 TTS backbone faster and smaller on a CPU-only
laptop, with a Marathi test case. Nothing here is production code.

## Goal

Run the Maya2 backbone (a Qwen3-based decoder, 28 layers, hidden size 2048) with lower RAM and good speed,
using ONNX Runtime on the CPU. Target: no swapping on a 16 GB machine.

## What was measured

Test: 28 backbone layers chained, 8-token prefill plus decode steps, on real Marathi token embeddings.
Accuracy is cosine similarity against the full-precision (fp32) chain.

| Variant | Accuracy (cos, worst step) | Chain wall, 14 steps | Decode ms/layer | Weights |
|---|---|---|---|---|
| fp32 | exact (per-layer max rel. err ~1e-6) | not timed | ~3.0 | ~5.6 GB |
| weight-only int8, accuracy_level 4, block 32 | 0.996 | 5.1 s | ~2.6 | ~1.4 GB |
| weight-only int8, default accuracy level, block 32 | 0.998 | 28.9 s | ~34 | ~1.4 GB |
| static int8 QDQ (activations quantized), MatMul only | 0.43 | 2.4 s | not measured | ~1.4 GB |
| dynamic int8 (activations quantized) | 0.96 to 0.51 (random inputs) | not measured | ~0.65 | ~1.4 GB |
| weight-only int4, block 32 | 0.77 at prefill, lower after | not measured | ~1.6 | ~0.7 GB |

## Findings

- Quantizing activations (dynamic or static int8) breaks accuracy over decode steps. Keeping activations in fp32
  and quantizing only weights works.
- In ONNX Runtime's MatMulNBits kernel, `accuracy_level=4` is what makes 8-bit weights fast on this CPU. The
  default level was about 13 times slower than fp32.
- Weight-only int4 is fastest but too inaccurate for speech.
- Per-layer ONNX export matches PyTorch to about 1e-6 in fp32, so the export itself is sound.

## Not yet done

- Full per-token timing with the depth predictor and Mimi decoder included.
- End-to-end audio quality check.
- Peak RAM check for a full session.

## Files

- `benchmarks/onnx_layer_spike.py`: export each decoder layer to ONNX, check against PyTorch, time decode.
- `benchmarks/onnx_chain_spike.py`: chain all layers, compare two variants on random inputs, weight-only quant.
- `benchmarks/onnx_static_spike.py`: calibrate and run static QDQ quantization, then compare chains on real text.
- `benchmarks/tts_quant_spike.py`, `benchmarks/tts_lowram_load.py`: earlier PyTorch weight-quant spike (int8/int4
  packed weights, low-RAM loader).
- `docs/2026-10-27-maya2-marathi-research-side-plan.md`: research plan.

Model weights, exported ONNX files, and generated audio are not included. Maya2 weights are third-party; download them
from their original source.
