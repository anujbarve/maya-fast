"""THROWAWAY spike: end-to-end Maya2 TTS with the backbone replaced by 28 ONNX Runtime layers (weight-only
int8, accuracy_level=4) plus the final RMSNorm. Everything else (depth, cb0 head, audio_in, Mimi, voice) stays
PyTorch fp32. The 28 PyTorch backbone layers are never built, so RAM stays well under the 16 GB limit.

    .venv-tts/bin/python benchmarks/maya_fast_e2e.py "<marathi text>" --onnx-dir <dir> -o out.wav
"""
import argparse
import gc
import os
import resource
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
N_LAYERS = 28
LAYER_PREFIX = "backbone.lm.model.layers."


def stub_lm(path, dtype, attn):
    """Backbone LM with zero decoder layers: embeddings + final norm + rotary only. Weights come later."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(path)
    cfg.num_hidden_layers = 0
    cfg.layer_types = []
    with torch.device("meta"):
        lm = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="sdpa")
    for m in lm.modules():
        if hasattr(m, "inv_freq") and hasattr(m, "rope_type"):
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            fn = m.rope_init_fn if getattr(m, "rope_init_fn", None) is not None else (
                m.compute_default_rope_parameters if m.rope_type == "default" else ROPE_INIT_FUNCTIONS[m.rope_type])
            with torch.device("cpu"):  # rotary tables are real tensors even inside the meta-device build
                inv, scale = fn(m.config, torch.device("cpu"))
            m.inv_freq = inv
            if hasattr(m, "original_inv_freq"):
                m.original_inv_freq = inv
            if hasattr(m, "attention_scaling"):
                m.attention_scaling = scale
    lm.tie_weights()
    return lm


def load_ort_layers(onnx_dir, suffix, threads=8, spin=True):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = False  # return freed memory to the OS instead of keeping an arena
    so.enable_mem_pattern = False
    if not spin:
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return [ort.InferenceSession(os.path.join(onnx_dir, f"layer_{i:02d}_{suffix}.onnx"), so,
                                 providers=["CPUExecutionProvider"]) for i in range(N_LAYERS)]


class OrtBackbone:
    """Drop-in for Backbone.forward: (inputs_embeds, past_key_values=..., use_cache=True) -> (hidden, past)."""

    def __init__(self, sessions, norm, rot, cfg):
        self.sessions, self.norm, self.rot = sessions, norm, rot
        self.stats = []  # (T, past, seconds) per call
        self.kvh = cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    def __call__(self, inputs_embeds, attention_mask=None, past_key_values=None, use_cache=False):
        from onnx_layer_spike import masks, rope, run_ort
        h = inputs_embeds.detach().float().numpy()
        T = h.shape[1]
        t_call = time.perf_counter()
        if past_key_values is None:
            z = np.zeros((1, self.kvh, 0, self.hd), np.float32)
            past_key_values = [(z, z) for _ in range(N_LAYERS)]
        past = past_key_values[0][0].shape[2]
        cos, sin = rope(self.rot, T, past, torch.float32)
        cos, sin = cos.numpy(), sin.numpy()
        mask = masks(T, past).numpy()
        new_state = []
        for sess, (pk, pv) in zip(self.sessions, past_key_values):
            r = run_ort(sess, {"h": h, "cos": cos, "sin": sin, "mask": mask, "past_k": pk, "past_v": pv})
            new_state.append((r["new_k"], r["new_v"]))
            h = r["out"]
        out = self.norm(torch.from_numpy(h))
        self.stats.append((T, past, time.perf_counter() - t_call))
        return out, new_state


def load_weights(model, paths, L):
    """Assign every checkpoint tensor (except the 28 decoder layers) into a meta-device model, one at a time.
    Nothing is randomly initialized and no weight is held twice, so the load peak is the model itself."""
    import tts_lowram_load as LR
    src = os.path.abspath(str(paths["ckpt"]))
    _, files, _aliases = L._safetensors_files(src)
    mods = dict(model.named_modules())
    seen, unexpected = set(), []
    for f in files:
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                if k.startswith(LAYER_PREFIX):
                    continue
                name = k[len(L.MODEL_PREFIX):] if k.startswith(L.MODEL_PREFIX) else k
                parent_name, _, leaf = name.rpartition(".")
                parent = mods.get(parent_name)
                if parent is None or (leaf not in parent._parameters and leaf not in parent._buffers):
                    unexpected.append(name)
                    continue
                is_param = leaf in parent._parameters
                t = fh.get_tensor(k).float()
                LR._set(model, name, t, is_param=is_param)
                seen.add(name)
                del t
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:5]}")
    # lm_head is tied to embed_tokens: re-tie after the embedding was replaced.
    model.backbone.lm.tie_weights()
    seen.add("backbone.lm.lm_head.weight")
    left = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    left = [n for n in left if not n.startswith(LAYER_PREFIX)]
    if left:
        raise RuntimeError(f"still on meta device (not in checkpoint): {left[:8]}")
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--suffix", default="a4")
    ap.add_argument("--q-serve", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ort-threads", type=int, default=8)
    ap.add_argument("--no-spin", action="store_true")
    ap.add_argument("--voice", default="Charvi")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="maya_fast.wav")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)

    import soundfile as sf
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from transformers import MimiModel
    import tts.model.backbone as BB
    import tts.model.load_s3 as L
    from maya3 import hub
    from maya3.engine import ROOT
    from maya3.prompt import DEFAULT_BUILDER, MODES, VoiceRegistry, load_named_voices
    from maya3 import guards as G
    from scripts import s2_serve_controls as S
    from tts.data.packing_v2 import PackerV2

    t0 = time.perf_counter()
    paths = hub.resolve("maya-research/maya2-native")
    cfg_dir = ROOT / "configs"
    config_path = str(cfg_dir / "s3_fresh_v1.yaml")
    mcfg = {"model": {"backbone": str(paths["backbone_dir"]), "attn": "sdpa"}, "data": {"max_tokens": 8192}}

    BB._lm_from_config = stub_lm  # backbone LM: embeddings + norm only, no decoder layers
    src = os.path.abspath(str(paths["ckpt"]))
    model_cfg, _ = L.resolve_model_cfg(src, cfg=mcfg, config=config_path)
    model_cfg["backbone"] = mcfg["model"]["backbone"]
    anchor_cfg = L._reconcile_anchor(model_cfg.get("speaker_anchor"),
                                     L.anchor_cfg_from_shapes(_shapes(paths, L)), require_anchor=True)
    with torch.device("meta"):  # shapes only; weights arrive from the checkpoint in load_weights
        model = L.build_eval_model(model_cfg, anchor_cfg, 1.0, bool(model_cfg.get("depth_grouped", False))).eval()
    print("RSS after_build mb=%.0f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20), flush=True)
    load_weights(model, paths, L)
    print("RSS after_weights mb=%.0f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20), flush=True)
    model.depth.float()

    rcfg = AutoConfig.from_pretrained(str(paths["backbone_dir"]))
    rcfg._attn_implementation = "sdpa"
    rot = Qwen3RotaryEmbedding(config=rcfg).eval()
    sessions = load_ort_layers(a.onnx_dir, a.suffix, a.ort_threads, not a.no_spin)
    print("RSS after_ort mb=%.0f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20), flush=True)
    backbone = OrtBackbone(sessions, model.backbone.lm.model.norm, rot, rcfg)
    model.backbone.forward = backbone

    packer = PackerV2(model.backbone.tokenizer, model.backbone.token_ids, max_tokens=8192)
    mimi = MimiModel.from_pretrained(str(paths["mimi_dir"])).eval()
    print("RSS after_mimi mb=%.0f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20), flush=True)
    prior = S.load_prior(str(cfg_dir / "text_frames_prior.json"))
    registry = VoiceRegistry()
    load_named_voices(registry, paths["voices_json"])
    voice = registry.get(a.voice)
    load_s = time.perf_counter() - t0
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS -> MB

    with torch.inference_mode():
        use_icl, use_mem = MODES[voice.mode]
        builder = voice.builder or DEFAULT_BUILDER
        voice.prefix = builder.prefix(packer, voice)
        voice.spk_refs = (np.asarray(voice.ecapa, np.float32).reshape(1, 192) if voice.ecapa is not None
                          else np.zeros((1, 192), np.float32))
        refs_t = torch.as_tensor(voice.spk_refs, device="cpu")[None]
        voice.mem = model.depth.spk_memory(refs_t, "ecapa", null_mask=torch.tensor([not use_mem], device="cpu"))
        voice.rmask = torch.ones(1, 1, dtype=torch.bool, device="cpu")

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
        prompt, n = S.prompt_from_example(ex, model, "cpu", torch.float32, q)
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
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    for T, past, s in backbone.stats[:3] + backbone.stats[-3:]:
        print(f"BB T={T} past={past} s={s:.3f}", flush=True)
    print(f"METRICS suffix={a.suffix} q={q} threads={a.threads} load_s={load_s:.1f} frames={codes.shape[-1]} "
          f"reason={reason} audio_s={audio_s:.2f} wall_s={wall:.1f} RTF={wall / max(audio_s, 1e-9):.2f} "
          f"peak_rss_after_load_mb={rss_after_load:.0f} peak_rss_mb={rss:.0f} "
          f"backbone_s={ph['backbone']:.1f} depth_s={ph['depth']:.1f} mimi_s={ph['mimi']:.1f}", flush=True)


def _shapes(paths, L):
    _, files, _ = L._safetensors_files(os.path.abspath(str(paths["ckpt"])))
    shapes = {}
    for f in files:
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                name = k[len(L.MODEL_PREFIX):] if k.startswith(L.MODEL_PREFIX) else k
                shapes[name] = torch.Size(fh.get_slice(k).get_shape())
    return shapes


if __name__ == "__main__":
    main()
