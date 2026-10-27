"""THROWAWAY spike: per-layer ONNX export of the Maya2 backbone (Qwen3 decoder), low-RAM.

Each decoder layer is built alone, loaded from safetensors one tensor at a time, exported to its own
ONNX file with KV cache I/O, checked against the PyTorch layer (prefill + one decode step), then
dynamic-int8 quantized. Peak RAM is one layer (~200 MB fp32), so nothing near swap.

    .venv-tts/bin/python benchmarks/onnx_layer_spike.py --out-dir <dir> --layers 0-27
"""
import argparse
import gc
import os
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

CKPT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--maya-research--maya2-native/snapshots/"
    "7cce38065147840d68011e18d25f4d68462e402f")
BACKBONE = os.path.join(CKPT, "backbone")
PREFIX = "backbone.lm.model.layers."


class LayerKV(nn.Module):
    """One decoder layer, KV cache as flat tensors, additive attention mask passed in."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, h, cos, sin, mask, past_k, past_v):
        cache = DynamicCache()
        cache.update(past_k, past_v, 0)
        out = self.layer(h, attention_mask=mask, position_ids=None, past_key_values=cache,
                         use_cache=True, cache_position=None, position_embeddings=(cos, sin))
        if isinstance(out, tuple):
            out = out[0]
        return out, cache.layers[0].keys, cache.layers[0].values


def load_layer(idx, cfg, shards):
    # layer_idx=0: each shell owns a one-slot DynamicCache (cache.update(..., 0)), so attention must use slot 0.
    # All layers are full_attention, so the index affects nothing else.
    layer = Qwen3DecoderLayer(cfg, layer_idx=0).float().eval()
    want = f"{PREFIX}{idx}."
    sd = {}
    for name, fh in shards:
        for k in fh.keys():
            if k.startswith(want):
                sd[k[len(want):]] = fh.get_tensor(k).float()
    missing, unexpected = layer.load_state_dict(sd, strict=True), None
    del sd
    return layer


def masks(T, past):
    """Additive causal mask (1, 1, T, past+T): query i sees keys j <= past + i."""
    m = torch.zeros(T, past + T)
    i = torch.arange(T)[:, None]
    j = torch.arange(past + T)[None, :]
    m = m.masked_fill(j > past + i, float("-inf"))
    return m[None, None]


def rope(rot, T, past, dtype):
    pos = torch.arange(past, past + T)[None]
    x = torch.zeros(1, T, 1, dtype=dtype)
    cos, sin = rot(x, pos)
    return cos, sin


def ort_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def run_ort(sess, feed):
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, feed)))


def export_layer(idx, layer, cfg, rot, out_dir, P, D):
    kvh = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    wrap = LayerKV(layer).eval()
    h0 = torch.randn(1, P, cfg.hidden_size) * 0.02
    cos, sin = rope(rot, P, 0, torch.float32)
    pk = torch.zeros(1, kvh, 0, hd)
    pv = torch.zeros(1, kvh, 0, hd)
    path = os.path.join(out_dir, f"layer_{idx:02d}_fp32.onnx")
    with torch.inference_mode():
        torch.onnx.export(
            wrap, (h0, cos, sin, masks(P, 0), pk, pv), path,
            input_names=["h", "cos", "sin", "mask", "past_k", "past_v"],
            output_names=["out", "new_k", "new_v"],
            dynamic_axes={"h": {1: "t"}, "cos": {1: "t"}, "sin": {1: "t"}, "mask": {2: "t", 3: "total"},
                          "past_k": {2: "past"}, "past_v": {2: "past"},
                          "out": {1: "t"}, "new_k": {2: "total"}, "new_v": {2: "total"}},
            opset_version=18, dynamo=False, do_constant_folding=True,
        )
    return path


def parity(idx, layer, cfg, rot, path, P):
    """PyTorch layer vs ORT: prefill P tokens, then one decode token with the returned cache."""
    kvh = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    sess = ort_session(path)
    torch.manual_seed(idx)
    h = torch.randn(1, P, cfg.hidden_size) * 0.02
    d = torch.randn(1, 1, cfg.hidden_size) * 0.02
    wrap = LayerKV(layer).eval()
    worst = 0.0
    with torch.inference_mode():
        empty = torch.zeros(1, kvh, 0, hd)
        cos, sin = rope(rot, P, 0, torch.float32)
        ref_o, rk, rv = wrap(h, cos, sin, masks(P, 0), empty, empty)
        got = run_ort(sess, {"h": h.numpy(), "cos": cos.numpy(), "sin": sin.numpy(),
                             "mask": masks(P, 0).numpy(), "past_k": empty.numpy(), "past_v": empty.numpy()})
        worst = max(worst, float(np.abs(got["out"] - ref_o.numpy()).max() / np.abs(ref_o.numpy()).max()))
        cos1, sin1 = rope(rot, 1, P, torch.float32)
        ref_d, rk2, rv2 = wrap(d, cos1, sin1, masks(1, P), rk, rv)
        got2 = run_ort(sess, {"h": d.numpy(), "cos": cos1.numpy(), "sin": sin1.numpy(),
                              "mask": masks(1, P).numpy(), "past_k": got["new_k"], "past_v": got["new_v"]})
        worst = max(worst, float(np.abs(got2["out"] - ref_d.numpy()).max() / np.abs(ref_d.numpy()).max()))
    return worst


def time_decode(path, cfg, rot, P, steps=20):
    kvh = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    sess = ort_session(path)
    pk = np.zeros((1, kvh, P, hd), np.float32)
    pv = np.zeros((1, kvh, P, hd), np.float32)
    d = np.random.randn(1, 1, cfg.hidden_size).astype(np.float32) * 0.02
    t = time.perf_counter()
    for s in range(steps):
        cos, sin = rope(rot, 1, P + s, torch.float32)
        r = run_ort(sess, {"h": d, "cos": cos.numpy(), "sin": sin.numpy(),
                           "mask": masks(1, P + s).numpy(), "past_k": pk, "past_v": pv})
        pk, pv = r["new_k"], r["new_v"]
    return (time.perf_counter() - t) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default="0")
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--quant", action="store_true", help="also dynamic-int8 quantize each layer")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_num_threads(8)

    lo, _, hi = a.layers.partition("-")
    layers = range(int(lo), int(hi or lo) + 1)

    cfg = AutoConfig.from_pretrained(BACKBONE)
    cfg._attn_implementation = "sdpa"
    rot = Qwen3RotaryEmbedding(config=cfg).eval()
    shards = []
    for f in sorted(os.listdir(CKPT)):
        if f.endswith(".safetensors") and "model" in f:
            real = os.path.realpath(os.path.join(CKPT, f))
            shards.append((f, safe_open(real, "pt")))

    for idx in layers:
        t0 = time.perf_counter()
        layer = load_layer(idx, cfg, shards)
        path = export_layer(idx, layer, cfg, rot, a.out_dir, a.prompt_len, None)
        err = parity(idx, layer, cfg, rot, path, a.prompt_len)
        line = f"LAYER {idx:02d} fp32 export_ok maxerr={err:.2e} decode_ms={time_decode(path, cfg, rot, a.prompt_len) * 1e3:.2f}"
        if a.quant:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            qpath = path.replace("_fp32.onnx", "_int8.onnx")
            quantize_dynamic(path, qpath, weight_type=QuantType.QInt8, per_channel=True, reduce_range=False)
            qerr = parity(idx, layer, cfg, rot, qpath, a.prompt_len)
            line += f" int8_maxerr={qerr:.2e} int8_decode_ms={time_decode(qpath, cfg, rot, a.prompt_len) * 1e3:.2f}"
        print(line + f" wall_s={time.perf_counter() - t0:.1f}", flush=True)
        del layer
        gc.collect()


if __name__ == "__main__":
    main()
