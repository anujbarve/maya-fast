"""THROWAWAY spike: run all 28 exported backbone layers in sequence, fp32 chain vs int8 chain, and report how
int8 error grows over prefill + decode steps. One ORT session alive at a time (RAM stays at one layer).

    .venv-tts/bin/python benchmarks/onnx_chain_spike.py --onnx-dir <dir> --decode-steps 16
"""
import argparse
import gc
import os

import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from onnx_layer_spike import BACKBONE, masks, ort_session, rope, run_ort

N_LAYERS = 28


def run_chain(onnx_dir, suffix, steps, cfg, rot, P):
    """steps[0]: (1, P, d) prefill; steps[s]: (1, 1, d) decode. Returns per-step outputs of the last layer."""
    kvh = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    seq = steps
    for idx in range(N_LAYERS):
        path = os.path.join(onnx_dir, f"layer_{idx:02d}_{suffix}.onnx")
        sess = ort_session(path)
        pk = np.zeros((1, kvh, 0, hd), np.float32)
        pv = np.zeros((1, kvh, 0, hd), np.float32)
        past = 0
        nxt = []
        for h in seq:
            T = h.shape[1]
            cos, sin = rope(rot, T, past, torch.float32)
            r = run_ort(sess, {"h": h, "cos": cos.numpy(), "sin": sin.numpy(),
                               "mask": masks(T, past).numpy(), "past_k": pk, "past_v": pv})
            pk, pv = r["new_k"], r["new_v"]
            past += T
            nxt.append(r["out"])
        seq = nxt
        del sess
        gc.collect()
    return seq


def make_wo8(onnx_dir, bits, block):
    """Weight-only quant of each fp32 layer (activations stay fp32), one layer in RAM at a time."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    for idx in range(N_LAYERS):
        src = os.path.join(onnx_dir, f"layer_{idx:02d}_fp32.onnx")
        dst = os.path.join(onnx_dir, f"layer_{idx:02d}_wo{bits}.onnx")
        q = MatMulNBitsQuantizer(onnx.load(src), bits=bits, block_size=block, is_symmetric=True)
        q.process()
        q.model.save_model_to_file(dst, False)
        del q
        gc.collect()
    print(f"WO{bits} wrote {N_LAYERS} layers block={block}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--make-wo", type=int, default=0, help="write weight-only quant layers at this bit width")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--suffix", default="int8", help="layer file suffix to test against fp32")
    a = ap.parse_args()

    if a.make_wo:
        make_wo8(a.onnx_dir, a.make_wo, a.block)
        return

    cfg = AutoConfig.from_pretrained(BACKBONE)
    cfg._attn_implementation = "sdpa"
    rot = Qwen3RotaryEmbedding(config=cfg).eval()

    rng = np.random.default_rng(a.seed)
    d = cfg.hidden_size
    steps = [(rng.standard_normal((1, a.prompt_len, d)) * 0.02).astype(np.float32)]
    steps += [(rng.standard_normal((1, 1, d)) * 0.02).astype(np.float32) for _ in range(a.decode_steps)]

    ref = run_chain(a.onnx_dir, "fp32", steps, cfg, rot, a.prompt_len)
    q = run_chain(a.onnx_dir, a.suffix, steps, cfg, rot, a.prompt_len)

    for s, (r, g) in enumerate(zip(ref, q)):
        rel = float(np.abs(g - r).max() / np.abs(r).max())
        cos = float(np.sum(g * r) / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-12))
        tag = "prefill" if s == 0 else f"decode{s}"
        print(f"STEP {tag:>9} rel_max={rel:.3e} cos={cos:.6f} ref_absmax={np.abs(r).max():.3f}", flush=True)


if __name__ == "__main__":
    main()
