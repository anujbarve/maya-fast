"""THROWAWAY spike: static int8 (QDQ) quantization of the 28 exported backbone layers, calibrated on real
Marathi token embeddings, one layer in RAM at a time. Compares against the fp32 chain on held-out text.

    .venv-tts/bin/python benchmarks/onnx_static_spike.py --onnx-dir <dir> --stage calib|quant|test
"""
import argparse
import gc
import os
import pickle
import time

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from onnx_chain_spike import N_LAYERS, run_chain
from onnx_layer_spike import BACKBONE, CKPT, masks, ort_session, rope, run_ort

CALIB_TEXT = ("महाराष्ट्रात पुणे हे एक मोठे शहर आहे. येथे अनेक शैक्षणिक संस्था आहेत. "
              "सकाळी लवकर उठून फिरायला जाणे आरोग्यासाठी चांगले असते. आज हवामान छान आहे.")
TEST_TEXT = "नमस्कार, मी तुमची मदत करण्यासाठी इथे आहे. आजचा दिवस खूप सुंदर आहे."


def embed_steps(text, tok, emb_w, n_decode=None):
    ids = tok(text).input_ids
    if n_decode is not None:
        ids = ids[: 8 + n_decode]
    e = emb_w[np.array(ids)]
    return [e[None, :8].astype(np.float32)] + [e[None, i:i + 1].astype(np.float32) for i in range(8, len(ids))]


def load_embed(idx=None):
    with safe_open(os.path.join(CKPT, "model-00001-of-00002.safetensors"), "pt") as fh:
        return fh.get_tensor("backbone.lm.model.embed_tokens.weight").float().numpy()


def capture(onnx_dir, steps, cfg, rot):
    """fp32 chain; record each layer's exact inputs per step (for calibration)."""
    kvh = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    seq, samples = steps, []
    for idx in range(N_LAYERS):
        sess = ort_session(os.path.join(onnx_dir, f"layer_{idx:02d}_fp32.onnx"))
        pk = np.zeros((1, kvh, 0, hd), np.float32)
        pv = np.zeros((1, kvh, 0, hd), np.float32)
        past, nxt, rec = 0, [], []
        for h in seq:
            T = h.shape[1]
            cos, sin = rope(rot, T, past, torch.float32)
            feed = {"h": h, "cos": cos.numpy(), "sin": sin.numpy(),
                    "mask": masks(T, past).numpy(), "past_k": pk, "past_v": pv}
            r = run_ort(sess, feed)
            rec.append(feed)
            pk, pv = r["new_k"], r["new_v"]
            past += T
            nxt.append(r["out"])
        samples.append(rec)
        seq = nxt
        del sess
        gc.collect()
    return samples


class Reader:
    def __init__(self, feeds):
        self.it = iter(feeds)

    def get_next(self):
        return next(self.it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--stage", required=True, choices=["calib", "quant", "test"])
    ap.add_argument("--calib-decode", type=int, default=48)
    ap.add_argument("--test-decode", type=int, default=16)
    ap.add_argument("--suffix", default="qdq")
    a = ap.parse_args()

    cfg = AutoConfig.from_pretrained(BACKBONE)
    cfg._attn_implementation = "sdpa"
    rot = Qwen3RotaryEmbedding(config=cfg).eval()
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    calib_pkl = os.path.join(a.onnx_dir, "calib.pkl")

    if a.stage == "calib":
        emb = load_embed()
        steps = embed_steps(CALIB_TEXT, tok, emb, a.calib_decode)
        del emb
        gc.collect()
        t = time.perf_counter()
        samples = capture(a.onnx_dir, steps, cfg, rot)
        with open(calib_pkl, "wb") as f:
            pickle.dump(samples, f)
        print(f"CALIB steps={len(steps)} s={time.perf_counter() - t:.1f}", flush=True)

    elif a.stage == "quant":
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
        with open(calib_pkl, "rb") as f:
            samples = pickle.load(f)
        for idx in range(N_LAYERS):
            src = os.path.join(a.onnx_dir, f"layer_{idx:02d}_fp32.onnx")
            dst = os.path.join(a.onnx_dir, f"layer_{idx:02d}_qdq.onnx")
            quantize_static(src, dst, Reader(samples[idx]), quant_format=QuantFormat.QDQ,
                            activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                            per_channel=True, op_types_to_quantize=["MatMul"])
            print(f"QDQ layer {idx:02d} written", flush=True)
            gc.collect()

    elif a.stage == "test":
        emb = load_embed()
        steps = embed_steps(TEST_TEXT, tok, emb, a.test_decode)
        del emb
        gc.collect()
        ref = run_chain(a.onnx_dir, "fp32", steps, cfg, rot, 8)
        t = time.perf_counter()
        q = run_chain(a.onnx_dir, a.suffix, steps, cfg, rot, 8)
        el = time.perf_counter() - t
        for s, (r, g) in enumerate(zip(ref, q)):
            rel = float(np.abs(g - r).max() / np.abs(r).max())
            cos = float(np.sum(g * r) / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-12))
            tag = "prefill" if s == 0 else f"decode{s}"
            print(f"STEP {tag:>9} rel_max={rel:.3e} cos={cos:.6f}", flush=True)
        print(f"QDQ chain wall_s={el:.2f} for {len(steps)} steps", flush=True)


if __name__ == "__main__":
    main()
