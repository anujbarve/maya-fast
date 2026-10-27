"""THROWAWAY spike: build Maya2 on the meta device and load safetensors one tensor at a time, packing every
backbone/depth nn.Linear to int8/int4 as it is read. Never materializes the full bf16 model in RAM.
Deleted in Task 3 with the rest of the spike code.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

GROUP = {4: 32, 8: 64}
SKIP = ("spk", "embed", "lm_head", "cb0", "audio_in", "anchor")  # kept unquantized, loaded as-is (anchor: .weight read directly)


def quant_pack(w, bits):
    group = GROUP[bits]
    out, inp = w.shape
    g = w.float().reshape(out, inp // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    levels = 15 if bits == 4 else 255
    scale = ((hi - lo) / levels).clamp(min=1e-8)
    q = ((g - lo) / scale).round().clamp(0, levels).to(torch.uint8).reshape(out, inp)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)) if bits == 4 else q
    return packed, scale.squeeze(-1).half(), lo.squeeze(-1).half()


def dequant(packed, scale, zero, bits, dtype):
    group = GROUP[bits]
    if bits == 4:
        q = torch.stack([(packed & 0x0F), (packed >> 4)], -1).flatten(1).float()
    else:
        q = packed.float()
    out, inp = q.shape
    w = q.reshape(out, inp // group, group) * scale.float()[..., None] + zero.float()[..., None]
    return w.reshape(out, inp).to(dtype)


class QLinear(nn.Module):
    def __init__(self, in_features, out_features, bits, packed, scale, zero, bias):
        super().__init__()
        self.bits = bits
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)
        self.register_buffer("zero", zero)
        self.bias = bias

    def forward(self, x):
        w = dequant(self.packed, self.scale, self.zero, self.bits, x.dtype)  # one layer, freed after the matmul
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, w, b)


def _set(root, dotted, value, is_param):
    parent_name, _, leaf = dotted.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if is_param:
        parent._parameters[leaf] = nn.Parameter(value, requires_grad=False)
    else:
        parent._buffers[leaf] = value


def load_lowram(paths, bits, cfg, config_path, build_eval_model, L):
    """paths: hub.resolve(...) dict; cfg: caller eval config; L: tts.model.load_s3 module.
    Returns the model in eval mode with backbone/depth Linears packed to `bits`."""
    src = os.path.abspath(str(paths["ckpt"]))
    _, files, aliases = L._safetensors_files(src)

    shapes, where = {}, {}
    for f in files:
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                name = k[len(L.MODEL_PREFIX):] if k.startswith(L.MODEL_PREFIX) else k
                shapes[name] = torch.Size(fh.get_slice(k).get_shape())
                where[name] = (f, k)

    model_cfg, _ = L.resolve_model_cfg(src, cfg=cfg, config=config_path)
    eval_model = (cfg or {}).get("model", {}) if cfg else {}
    for k in ("backbone", "attn"):
        if k in eval_model:
            model_cfg[k] = eval_model[k]
    anchor_cfg = L._reconcile_anchor(model_cfg.get("speaker_anchor"), L.anchor_cfg_from_shapes(shapes),
                                     require_anchor=True)
    depth_grouped = bool(model_cfg.get("depth_grouped", False))

    # No outer meta context: Backbone already builds its LM uninitialized (to_empty) and recomputes RoPE itself,
    # and a surrounding meta context breaks that recompute. DepthV2 is randomly initialized (~2.2 GB fp32 touched).
    import resource
    rss = lambda tag: print(f"LOWRAM {tag} maxrss_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f}", flush=True)
    rss("before build")
    model = build_eval_model(model_cfg, anchor_cfg, 1.0, depth_grouped)
    rss("after build")
    model.eval()

    # Pass 1: swap the packable Linears for QLinear shells (weights arrive in pass 2).
    packable = []
    for pname, parent in list(model.named_modules()):
        for cname, child in list(parent.named_children()):
            full = f"{pname}.{cname}" if pname else cname
            if isinstance(child, nn.Linear) and not any(k in full for k in SKIP) and child.in_features % GROUP[bits] == 0:
                packable.append((full, parent, cname, child))

    # Pass 2: read tensors one at a time, quantize packable Linear weights, assign everything else.
    def read(name):
        # open/close per tensor so the mmap pages are released after each read (safe_open keeps them resident)
        f, k = where[name]
        with safe_open(f, "pt") as fh:
            return fh.get_tensor(k)

    pack_names = set()
    for full, parent, cname, child in packable:
        wname = f"{full}.weight"
        if wname not in where:
            continue
        w = read(wname)
        packed, scale, zero = quant_pack(w, bits)
        del w
        bias_name = f"{full}.bias"
        bias = nn.Parameter(read(bias_name), requires_grad=False) if bias_name in where else None
        q = QLinear(child.in_features, child.out_features, bits, packed, scale, zero, bias)
        setattr(parent, cname, q)
        pack_names.add(wname)
        pack_names.add(bias_name)

    params = dict(model.named_parameters())
    bufs = dict(model.named_buffers())
    unexpected = []
    for name in where:
        if name in pack_names or name.startswith(L.ANCHOR_PREFIX):
            continue
        t = read(name)
        if name in params:
            _set(model, name, nn.Parameter(t, requires_grad=False), is_param=True)
        elif name in bufs:
            _set(model, name, t, is_param=False)
        else:
            unexpected.append(name)
    rss("after pass2")
    if unexpected:
        print(f"LOWRAM unexpected={len(unexpected)} {unexpected[:5]}", flush=True)

    # Tied weights: point each alias at its stored tensor, as _load_safetensors does.
    if aliases:
        for alias, target in aliases.items():
            a = alias[len(L.MODEL_PREFIX):] if alias.startswith(L.MODEL_PREFIX) else alias
            t = target[len(L.MODEL_PREFIX):] if target.startswith(L.MODEL_PREFIX) else target
            if a in dict(model.named_parameters()) and t in dict(model.named_parameters()):
                _set(model, a, dict(model.named_parameters())[t], is_param=True)

    left = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    return model, left
