# maya-fast

Research notes and throwaway spike scripts for making the Maya2 TTS backbone faster and smaller on a CPU-only
laptop, with a Marathi test case. Nothing here is production code.

## Goal

Run the Maya2 backbone (a Qwen3-based decoder, 28 layers, hidden size 2048) with lower RAM and good speed,
using ONNX Runtime on the CPU. Target: no swapping on a 16 GB machine.

## Layer-level results

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

## End-to-end results

Full Maya2 pipeline (speaker voice, depth predictor, cb0 head, Mimi decoder) with the backbone replaced by the
28 weight-only int8 (accuracy_level 4) ONNX layers plus the final RMSNorm. The depth predictor, cb0 head and Mimi
stay in PyTorch fp32. Test sentence: "नमस्कार, मी तुमची मदत करण्यासाठी इथे आहे." (3.12 s of audio, 8 threads).

| Run | RTF | Backbone | Depth | Peak RSS |
|---|---|---|---|---|
| Earlier PyTorch int4 (all-PyTorch) | 48.5 | 85 s | 54 s | 4.2 GB |
| ORT int8, default ORT threads | 17 | 25 s | 14 s | 6.0 GB |
| ORT int8, spinning disabled (`--no-spin`) | 5.3 to 5.6 | 2.4 to 2.9 s | 13 to 14 s | 7.3 GB |

Notes:
- Disabling ORT thread spinning cut backbone time about 10 times. ORT and PyTorch threads were competing for cores.
- Depth is now the bottleneck: about 0.35 s per frame in PyTorch fp32.
- Peak RSS of 7.3 GB occurs while loading, not during inference. The depth predictor is randomly initialized
  (about 2.2 GB) before the checkpoint is copied in.
- Swap was already at 2.7 GB used before the runs (other processes). The runs did not increase it.
- Audio quality has not been verified by ear. Output files exist but have only been checked by metrics.

## Findings

- Quantizing activations (dynamic or static int8) breaks accuracy over decode steps. Keeping activations in fp32
  and quantizing only weights works.
- In ONNX Runtime's MatMulNBits kernel, `accuracy_level=4` is what makes 8-bit weights fast on this CPU. The
  default level was about 13 times slower than fp32.
- Weight-only int4 is fastest but too inaccurate for speech.
- Per-layer ONNX export matches PyTorch to about 1e-6 in fp32, so the export itself is sound.
- ORT thread spinning can dominate latency when another thread pool (PyTorch) is active. Disable it in mixed setups.

## Not yet done

- Listening test of the end-to-end output against the earlier int4 output.
- Reducing peak RAM at load time (avoid random depth init, or load depth weights first).
- Moving the depth predictor to ORT.
- Full session RAM check over longer utterances.

## Files

- `benchmarks/onnx_layer_spike.py`: export each decoder layer to ONNX, check against PyTorch, time decode.
- `benchmarks/onnx_chain_spike.py`: chain all layers, compare two variants on random inputs, weight-only quant.
- `benchmarks/onnx_static_spike.py`: calibrate and run static QDQ quantization, then compare chains on real text.
- `benchmarks/maya_fast_e2e.py`: end-to-end Maya2 TTS with the ORT backbone. Options: `--onnx-dir`, `--suffix`,
  `--ort-threads`, `--no-spin`.
- `benchmarks/tts_quant_spike.py`, `benchmarks/tts_lowram_load.py`: earlier PyTorch weight-quant spike (int8/int4
  packed weights, low-RAM loader).
- `docs/2026-10-27-maya2-marathi-research-side-plan.md`: research plan.

Model weights, exported ONNX files, and generated audio are not included. Maya2 weights are third-party; download them
from their original source.
