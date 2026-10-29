"""THROWAWAY spike: run Maya2 (maya-research/maya2-native) on CPU and measure it. Deleted in Task 3.

    .venv-tts/bin/python benchmarks/tts_spike.py "<marathi text>" --q-serve 16 --dtype fp32 --threads 8 -o out.wav

Mirrors maya3.engine.Engine.__init__ / prepare_voice / _plan minus the CUDA parts (no StaticBackbone, FastDepth, graphs).
"""
import argparse
import os
import resource
import sys
import time

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--q-serve", type=int, default=16)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
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
    from maya3.engine import ROOT  # sys.path gets <root> and <root>/scripts
    from scripts import s2_serve_controls as S
    from tts.data.packing_v2 import PackerV2
    from tts.model.load_s3 import load_model_for_eval
    from maya3 import guards as G

    # in a pip install, configs/ is a top-level package in site-packages
    cfg_dir = ROOT / "configs"

    t0 = time.perf_counter()
    paths = hub.resolve("maya-research/maya2-native")
    mcfg = {"model": {"backbone": str(paths["backbone_dir"]), "attn": "sdpa"}, "data": {"max_tokens": 8192}}
    model, _prov = load_model_for_eval(str(paths["ckpt"]), device="cpu", dtype=dtype, cfg=mcfg,
                                       config=str(cfg_dir / "s3_fresh_v1.yaml"),
                                       amortize_frac=1.0, require_anchor=True, verbose=False)
    model.eval()
    model.depth.float()  # as Engine.__init__ does: backbone in bf16, depth stack stays fp32
    packer = PackerV2(model.backbone.tokenizer, model.backbone.token_ids, max_tokens=8192)
    mimi = MimiModel.from_pretrained(str(paths["mimi_dir"])).eval()
    prior = S.load_prior(str(cfg_dir / "text_frames_prior.json"))
    registry = VoiceRegistry()
    load_named_voices(registry, paths["voices_json"])
    voice = registry.get(a.voice)

    # prepare_voice (no KV caching)
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

    # per-phase timers
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
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # bytes on macOS
    print(f"METRICS dtype={a.dtype} q={q} threads={a.threads} load_s={load_s:.1f} frames={codes.shape[-1]} reason={reason} "
          f"prompt_tokens={n} audio_s={audio_s:.2f} wall_s={wall:.1f} RTF={wall / max(audio_s, 1e-9):.2f} "
          f"peak_rss_mb={rss_mb:.0f} backbone_s={ph['backbone']:.1f} depth_s={ph['depth']:.1f} mimi_s={ph['mimi']:.1f}")


if __name__ == "__main__":
    main()
