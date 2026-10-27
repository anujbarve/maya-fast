"""THROWAWAY spike: tts_spike.py with low-RAM weight quantization. Deleted in Task 3.

    .venv-tts/bin/python benchmarks/tts_quant_spike.py "<marathi text>" --quant int4 -o int4.wav

Every nn.Linear in the backbone and depth is replaced by QLinear: packed int8/int4 weights stay resident,
and each forward dequantizes ONE layer to the activation dtype, multiplies, and drops the temporary.
Asymmetric (min/max + zero) per-group quantization. int4 group 32, int8 group 64.
"""
import argparse
import os
import resource
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
SKIP = ("spk", "embed", "lm_head", "cb0", "audio_in")  # kept bf16: speaker path, embeddings, cb0 head


def quant_pack(w, bits, group):
    out, inp = w.shape
    g = w.float().reshape(out, inp // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    levels = 15 if bits == 4 else 255
    scale = ((hi - lo) / levels).clamp(min=1e-8)
    q = ((g - lo) / scale).round().clamp(0, levels).to(torch.uint8).reshape(out, inp)
    if bits == 4:
        packed = q[:, 0::2] | (q[:, 1::2] << 4)
    else:
        packed = q
    return packed, scale.squeeze(-1).half(), lo.squeeze(-1).half()


def dequant(packed, scale, zero, bits, group, dtype):
    if bits == 4:
        q = torch.stack([(packed & 0x0F), (packed >> 4)], -1).flatten(1).float()
    else:
        q = packed.float()
    out, inp = q.shape
    w = q.reshape(out, inp // group, group) * scale.float()[..., None] + zero.float()[..., None]
    return w.reshape(out, inp).to(dtype)


class QLinear(nn.Module):
    def __init__(self, lin, bits, group):
        super().__init__()
        self.bits, self.group = bits, group
        self.in_features, self.out_features = lin.in_features, lin.out_features
        packed, scale, zero = quant_pack(lin.weight.data, bits, group)
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)
        self.register_buffer("zero", zero)
        self.bias = lin.bias

    def forward(self, x):
        w = dequant(self.packed, self.scale, self.zero, self.bits, self.group, x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, w, b)


def apply_quant(model, mode):
    bits, group = (4, 32) if mode == "int4" else (8, 64)
    n, skipped = 0, []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}"
            if not isinstance(child, nn.Linear):
                continue
            if any(k in full for k in SKIP) or child.in_features % group:
                skipped.append(full)
                continue
            setattr(parent, child_name, QLinear(child, bits, group))
            n += 1
    weight_bytes = sum(b.numel() * b.element_size() for m in model.modules() if isinstance(m, QLinear)
                       for b in m.buffers())
    print(f"QUANT {mode} converted={n} kept={len(skipped)} weight_mb={weight_bytes / 2**20:.0f}", flush=True)
    return weight_bytes / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--q-serve", type=int, default=16)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--quant", choices=["none", "int8", "int4"], default="none")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--voice", default="Charvi")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="out.wav")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]

    import soundfile as sf
    from transformers import MimiModel
    import maya3
    from maya3 import hub
    from maya3.prompt import DEFAULT_BUILDER, MODES, VoiceRegistry, load_named_voices
    from maya3.engine import ROOT
    from scripts import s2_serve_controls as S
    from tts.data.packing_v2 import PackerV2
    from tts.model.load_s3 import load_model_for_eval
    from maya3 import guards as G

    cfg_dir = ROOT / "configs"

    t0 = time.perf_counter()
    paths = hub.resolve("maya-research/maya2-native")
    mcfg = {"model": {"backbone": str(paths["backbone_dir"]), "attn": "sdpa"}, "data": {"max_tokens": 8192}}
    if a.quant == "none":
        model, _prov = load_model_for_eval(str(paths["ckpt"]), device="cpu", dtype=dtype, cfg=mcfg,
                                           config=str(cfg_dir / "s3_fresh_v1.yaml"),
                                           amortize_frac=1.0, require_anchor=True, verbose=False)
        model.eval()
        model.depth.float()
        weights_mb = float("nan")
    else:
        import tts_lowram_load as LR
        from tts.model import load_s3 as L
        bits = 4 if a.quant == "int4" else 8
        model, left = LR.load_lowram(paths, bits, mcfg, str(cfg_dir / "s3_fresh_v1.yaml"), L.build_eval_model, L)
        if left:
            raise RuntimeError(f"meta tensors left after load: {left[:10]}")
        model.depth.float()
        weights_mb = sum(b.numel() * b.element_size() for m in model.modules() if isinstance(m, LR.QLinear)
                         for b in m.buffers()) / 2**20
        print(f"QUANT {a.quant} weights_mb={weights_mb:.0f}", flush=True)
    packer = PackerV2(model.backbone.tokenizer, model.backbone.token_ids, max_tokens=8192)
    mimi = MimiModel.from_pretrained(str(paths["mimi_dir"])).eval()
    prior = S.load_prior(str(cfg_dir / "text_frames_prior.json"))
    registry = VoiceRegistry()
    load_named_voices(registry, paths["voices_json"])
    voice = registry.get(a.voice)

    with torch.inference_mode():
        use_icl, use_mem = MODES[voice.mode]
        builder = voice.builder or DEFAULT_BUILDER
        voice.prefix = builder.prefix(packer, voice)
        voice.spk_refs = (np.asarray(voice.ecapa, np.float32).reshape(1, 192) if voice.ecapa is not None
                          else np.zeros((1, 192), np.float32))
        refs_t = torch.as_tensor(voice.spk_refs, device="cpu")[None]
        voice.mem = model.depth.spk_memory(refs_t, "ecapa", null_mask=torch.tensor([not use_mem], device="cpu"))
        voice.rmask = torch.ones(1, 1, dtype=torch.bool, device="cpu")
    load_s = time.perf_counter() - t0
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    ph = {"backbone": 0.0, "depth": 0.0, "mimi": 0.0}

    def timed(key, fn):
        def w(*args, **kw):
            t = time.perf_counter()
            try:
                return fn(*args, **kw)
            finally:
                ph[key] += time.perf_counter() - t
        return w

    S.ServeSession._step = timed("backbone", S.ServeSession._step)
    S.depth_generate = timed("depth", S.depth_generate)
    mimi.decode = timed("mimi", mimi.decode)

    text, lang = a.text.strip(), "mr"
    q = a.q_serve
    t_wall = time.perf_counter()
    with torch.inference_mode():
        ex = builder.example(packer, voice, text, lang)
        prompt, n = S.prompt_from_example(ex, model, "cpu", dtype, q)
        prompt = prompt.to(dtype)
        ef, er = S.expected_frames(prior, lang, len(text))
        g = G.GuardConfig()
        dur = G.DurationPrior()
        loose = min(1500, int(2.0 * (ef + er) + 40))
        cap = loose
        if g.enabled:
            cap = min(cap, max(int(ef * 0.5) + 8, dur.cap_frames(text, lang, g.dur_k, g.dur_margin_s)))
        minimum = max(4, int(0.5 * ef))
        sess = S.ServeSession(model, voice.mem, voice.rmask, q_serve=q, device="cpu")
        codes, reason, _ = sess.speak(prompt, cap, stop="controlled", min_frames=minimum)
        wav = mimi.decode(codes[None].long())[0][0, 0].float().cpu().numpy()
    wall = time.perf_counter() - t_wall

    sf.write(a.out, wav, 24000)
    audio_s = len(wav) / 24000
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(f"METRICS dtype={a.dtype} quant={a.quant} weights_mb={weights_mb:.0f} q={q} threads={a.threads} "
          f"load_s={load_s:.1f} frames={codes.shape[-1]} reason={reason} audio_s={audio_s:.2f} wall_s={wall:.1f} "
          f"RTF={wall / max(audio_s, 1e-9):.2f} peak_rss_after_load_mb={rss_after_load:.0f} peak_rss_mb={rss_mb:.0f} "
          f"backbone_s={ph['backbone']:.1f} depth_s={ph['depth']:.1f} mimi_s={ph['mimi']:.1f}", flush=True)


if __name__ == "__main__":
    main()
