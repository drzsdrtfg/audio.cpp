"""Convert the official Smart Turn v3 ONNX model into audio.cpp native weights.

The official releases (https://huggingface.co/pipecat-ai/smart-turn-v3) ship ONNX
exports of the Whisper-Tiny-encoder + attention-pool + classifier model. The ONNX
graph keeps most encoder matmul weights as anonymous constants, so this script
traces the graph to recover every tensor, renames tensors to the audio.cpp
HF-encoder layout, normalizes linear weights to [out_features, in_features], and
writes:

    <output_dir>/smart_turn.safetensors
    <output_dir>/smart_turn_config.json

Usage:
    python scripts/convert_smart_turn.py \
        --onnx smart-turn-v3.2-gpu.onnx \
        --output models/smart_turn \
        [--fixtures models/smart-turn/fixtures]   # optional parity self-check

The optional self-check re-implements the forward pass in numpy using the
extracted tensors and compares against onnxruntime logits for every fixture in
the fixtures directory (fixtures created with models/smart-turn tooling).
"""

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


def trace_weights(model_path):
    model = onnx.load(model_path)
    graph = model.graph
    inits = {t.name: numpy_helper.to_array(t) for t in graph.initializer}

    producers = {}
    for node in graph.node:
        for out in node.output:
            producers[out] = node

    def matmul_weight_name_for_bias(bias_name):
        """Find the MatMul whose result is bias-added; return its initializer weight name."""
        for node in graph.node:
            if node.op_type != "Add":
                continue
            if bias_name not in node.input:
                continue
            other = node.input[0] if node.input[1] == bias_name else node.input[1]
            up = producers.get(other)
            if up is None or up.op_type != "MatMul":
                raise RuntimeError(f"bias {bias_name} is not fed by a MatMul")
            for cand in up.input:
                if cand != other and cand in inits:
                    return cand
            raise RuntimeError(f"no initializer weight found for {bias_name}")
        raise RuntimeError(f"bias {bias_name} not found in graph")

    def matmul_input_tensor_for_bias(bias_name):
        """Return the activation tensor feeding the MatMul that bias_name is added to."""
        for node in graph.node:
            if node.op_type != "Add" or bias_name not in node.input:
                continue
            other = node.input[0] if node.input[1] == bias_name else node.input[1]
            up = producers.get(other)
            for cand in up.input:
                if cand != other and cand not in inits:
                    return cand
        raise RuntimeError(f"activation input for {bias_name} not found")

    def find_k_weight_name(q_weight_name, activation_tensor):
        """The key projection has no bias: it is the remaining MatMul over the
        same activation tensor whose weight is not q/v/out."""
        taken = {q_weight_name}
        for node in graph.node:
            if node.op_type != "MatMul" or activation_tensor not in node.input:
                continue
            for cand in node.input:
                if cand != activation_tensor and cand in inits and cand not in taken:
                    return cand
        raise RuntimeError("k_proj weight not found")

    out = {}

    # ---- encoder frontend (named in the ONNX graph) ----
    out["model.encoder.conv1.weight"] = inits["inner.encoder.conv1.weight"]
    out["model.encoder.conv1.bias"] = inits["inner.encoder.conv1.bias"]
    out["model.encoder.conv2.weight"] = inits["inner.encoder.conv2.weight"]
    out["model.encoder.conv2.bias"] = inits["inner.encoder.conv2.bias"]
    out["model.encoder.embed_positions.weight"] = inits["inner.encoder.embed_positions.weight"]

    layers = 0
    while f"inner.encoder.layers.{layers}.final_layer_norm.weight" in inits:
        layers += 1

    for i in range(layers):
        p = f"inner.encoder.layers.{i}"
        dst = f"model.encoder.layers.{i}"
        out[f"{dst}.self_attn_layer_norm.weight"] = inits[f"{p}.self_attn_layer_norm.weight"]
        out[f"{dst}.self_attn_layer_norm.bias"] = inits[f"{p}.self_attn_layer_norm.bias"]
        out[f"{dst}.final_layer_norm.weight"] = inits[f"{p}.final_layer_norm.weight"]
        out[f"{dst}.final_layer_norm.bias"] = inits[f"{p}.final_layer_norm.bias"]
        out[f"{dst}.fc1.bias"] = inits[f"{p}.fc1.bias"]
        out[f"{dst}.fc2.bias"] = inits[f"{p}.fc2.bias"]

        q = inits[matmul_weight_name_for_bias(f"{p}.self_attn.q_proj.bias")]
        v = inits[matmul_weight_name_for_bias(f"{p}.self_attn.v_proj.bias")]
        o = inits[matmul_weight_name_for_bias(f"{p}.self_attn.out_proj.bias")]
        k = inits[find_k_weight_name(
            matmul_weight_name_for_bias(f"{p}.self_attn.q_proj.bias"),
            matmul_input_tensor_for_bias(f"{p}.self_attn.q_proj.bias"))]
        # MatMul layout is y = x @ W  ->  transpose to torch [out, in].
        out[f"{dst}.self_attn.q_proj.weight"] = q.T
        out[f"{dst}.self_attn.k_proj.weight"] = k.T
        out[f"{dst}.self_attn.v_proj.weight"] = v.T
        out[f"{dst}.self_attn.out_proj.weight"] = o.T
        out[f"{dst}.self_attn.q_proj.bias"] = inits[f"{p}.self_attn.q_proj.bias"]
        out[f"{dst}.self_attn.v_proj.bias"] = inits[f"{p}.self_attn.v_proj.bias"]
        out[f"{dst}.self_attn.out_proj.bias"] = inits[f"{p}.self_attn.out_proj.bias"]
        out[f"{dst}.fc1.weight"] = inits[matmul_weight_name_for_bias(f"{p}.fc1.bias")].T
        out[f"{dst}.fc2.weight"] = inits[matmul_weight_name_for_bias(f"{p}.fc2.bias")].T

    out["model.encoder.layer_norm.weight"] = inits["inner.encoder.layer_norm.weight"]
    out["model.encoder.layer_norm.bias"] = inits["inner.encoder.layer_norm.bias"]

    # ---- heads ----
    # pool_attention runs as plain MatMul (y = x @ W): transpose to [out, in].
    out["pool_attention.0.weight"] = inits[matmul_weight_name_for_bias("inner.pool_attention.0.bias")].T
    out["pool_attention.0.bias"] = inits["inner.pool_attention.0.bias"]
    out["pool_attention.2.weight"] = inits[matmul_weight_name_for_bias("inner.pool_attention.2.bias")].T
    out["pool_attention.2.bias"] = inits["inner.pool_attention.2.bias"]

    # classifier is exported with Gemm(transB=1): weights are already [out, in].
    for idx in ("0", "1", "4", "6"):
        out[f"classifier.{idx}.weight"] = inits[f"inner.classifier.{idx}.weight"]
        out[f"classifier.{idx}.bias"] = inits[f"inner.classifier.{idx}.bias"]

    # ---- attribute-derived config ----
    ln_eps = 1.0e-5
    softmax_axis = None
    reduce_axes = None
    for node in graph.node:
        if node.op_type == "LayerNormalization":
            for attr in node.attribute:
                if attr.name == "epsilon":
                    ln_eps = min(ln_eps, attr.f) if ln_eps else attr.f
        if node.name == "softmax" or (node.op_type == "Softmax" and "pool_attention" in ""):
            for attr in node.attribute:
                if attr.name == "axis":
                    softmax_axis = attr.i
        if node.op_type == "ReduceSum":
            for attr in node.attribute:
                if attr.name == "axes":
                    reduce_axes = list(attr.ints)
    # The final softmax before the weighted sum is the last Softmax node.
    last_softmax_axis = None
    for node in graph.node:
        if node.op_type == "Softmax":
            for attr in node.attribute:
                if attr.name == "axis":
                    last_softmax_axis = attr.i
    reduce_axis = reduce_axes[0] if reduce_axes else 1

    config = {
        "sample_rate": 16000,
        "chunk_seconds": 8,
        "chunk_samples": 128000,
        "n_mels": 80,
        "n_fft": 400,
        "hop_length": 160,
        "mel_frames": 800,
        "n_audio_ctx": int(out["model.encoder.embed_positions.weight"].shape[0]),
        "d_model": int(out["model.encoder.conv1.bias"].shape[0]),
        "n_audio_head": 6,
        "n_audio_layer": layers,
        "ffn_dim": int(out["model.encoder.layers.0.fc1.bias"].shape[0]),
        "layer_norm_eps": float(ln_eps),
        "softmax_axis": int(last_softmax_axis if last_softmax_axis is not None else 1),
        "pool_reduce_axis": int(reduce_axis),
        "threshold": 0.5,
    }
    return out, config


# --------------------------------------------------------------------------
# numpy reference forward pass (converter self-check)
# --------------------------------------------------------------------------

def gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def layer_norm(x, w, b, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def conv1d(x, w, b, stride, pad):
    # x: [T, C_in], w: [C_out, C_in, K], PyTorch Conv1d semantics (zero padding)
    if pad > 0:
        x = np.pad(x, ((pad, pad), (0, 0)))
    c_out, c_in, k = w.shape
    t_out = (x.shape[0] - k) // stride + 1
    cols = np.stack([x[i * stride:i * stride + k] for i in range(t_out)])  # [T_out, K, C_in]
    y = np.einsum("tkc,ock->to", cols, w) + b
    return y


def attention(x, wq, wk, wv, wo, bq, bo, bv, heads, eps_scale):
    t, d = x.shape
    hd = d // heads
    q = (x @ wq.T + bq).reshape(t, heads, hd).transpose(1, 0, 2)
    k = (x @ wk.T).reshape(t, heads, hd).transpose(1, 0, 2)
    v = (x @ wv.T + bv).reshape(t, heads, hd).transpose(1, 0, 2)
    scores = q @ k.transpose(0, 2, 1) * eps_scale
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn /= attn.sum(axis=-1, keepdims=True)
    ctx = (attn @ v).transpose(1, 0, 2).reshape(t, d)
    return ctx @ wo.T + bo


def numpy_forward(weights, config, mel):
    eps = config["layer_norm_eps"]
    d = config["d_model"]
    heads = config["n_audio_head"]
    x = gelu(conv1d(mel.T, weights["model.encoder.conv1.weight"], weights["model.encoder.conv1.bias"], 1, 1))
    x = gelu(conv1d(x, weights["model.encoder.conv2.weight"], weights["model.encoder.conv2.bias"], 2, 1))
    x = x + weights["model.encoder.embed_positions.weight"]
    for i in range(config["n_audio_layer"]):
        p = f"model.encoder.layers.{i}"
        h = layer_norm(x, weights[f"{p}.self_attn_layer_norm.weight"], weights[f"{p}.self_attn_layer_norm.bias"], eps)
        a = attention(
            h,
            weights[f"{p}.self_attn.q_proj.weight"], weights[f"{p}.self_attn.k_proj.weight"],
            weights[f"{p}.self_attn.v_proj.weight"], weights[f"{p}.self_attn.out_proj.weight"],
            weights[f"{p}.self_attn.q_proj.bias"], weights[f"{p}.self_attn.out_proj.bias"],
            weights[f"{p}.self_attn.v_proj.bias"],
            heads, 1.0 / np.sqrt(d // heads))
        x = x + a
        h = layer_norm(x, weights[f"{p}.final_layer_norm.weight"], weights[f"{p}.final_layer_norm.bias"], eps)
        x = x + gelu(h @ weights[f"{p}.fc1.weight"].T + weights[f"{p}.fc1.bias"]) @ weights[f"{p}.fc2.weight"].T + weights[f"{p}.fc2.bias"]
    x = layer_norm(x, weights["model.encoder.layer_norm.weight"], weights["model.encoder.layer_norm.bias"], eps)
    pool_logits = ((np.tanh(x @ weights["pool_attention.0.weight"].T + weights["pool_attention.0.bias"]))
                   @ weights["pool_attention.2.weight"].T + weights["pool_attention.2.bias"])
    # pool_logits: [positions, 1] in this 2-D layout; softmax over the position axis.
    pool_logits = pool_logits - pool_logits.max(axis=0, keepdims=True)
    attn_w = np.exp(pool_logits)
    attn_w /= attn_w.sum(axis=0, keepdims=True)
    pooled = (x * attn_w).sum(axis=0)
    h = pooled @ weights["classifier.0.weight"].T + weights["classifier.0.bias"]
    h = layer_norm(h, weights["classifier.1.weight"], weights["classifier.1.bias"], eps)
    h = gelu(h)
    h = h @ weights["classifier.4.weight"].T + weights["classifier.4.bias"]
    h = gelu(h)
    logits = h @ weights["classifier.6.weight"].T + weights["classifier.6.bias"]
    return 1.0 / (1.0 + np.exp(-logits))


def self_check(weights, config, fixtures_dir):
    import onnxruntime as ort
    import soundfile as sf

    so = ort.SessionOptions()
    session = ort.InferenceSession(
        os.path.join(fixtures_dir, "..", "smart-turn-v3.2-gpu.onnx"),
        sess_options=so, providers=["CPUExecutionProvider"])
    meta = json.load(open(os.path.join(fixtures_dir, "expected.json")))
    worst = 0.0
    for entry in meta["fixtures"]:
        audio, sr = sf.read(os.path.join(fixtures_dir, entry["name"] + ".wav"), dtype="float32")
        mel = np.load(os.path.join(fixtures_dir, entry["name"] + ".hf_mel.npy"))
        ort_prob = float(np.asarray(
            session.run(None, {"input_features": mel[None, ...]})[0]).ravel()[0])
        mine = float(numpy_forward(weights, config, mel).ravel()[0])
        diff = abs(mine - ort_prob)
        worst = max(worst, diff)
        status = "OK " if diff < 1e-3 else "FAIL"
        print(f"  [{status}] {entry['name']}: ort={ort_prob:.6f} numpy={mine:.6f} |d|={diff:.2e}")
    if worst >= 1e-3:
        raise SystemExit(f"self-check FAILED (worst |d| = {worst:.2e})")
    print(f"self-check passed, worst |d| = {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fixtures", default=None, help="run the numpy-vs-ort parity self-check")
    args = ap.parse_args()

    print(f"tracing {args.onnx} ...")
    weights, config = trace_weights(args.onnx)
    print(f"extracted {len(weights)} tensors; layers={config['n_audio_layer']} d_model={config['d_model']} heads={config['n_audio_head']}")

    # sanity-check shapes against expectations
    assert weights["model.encoder.conv1.weight"].shape == (config["d_model"], 80, 3)
    assert weights["model.encoder.embed_positions.weight"].shape[1] == config["d_model"]
    assert config["n_audio_ctx"] * 2 == config["mel_frames"]
    assert weights["pool_attention.0.weight"].shape == (256, config["d_model"])
    assert weights["classifier.0.weight"].shape == (256, config["d_model"])

    os.makedirs(args.output, exist_ok=True)
    save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in weights.items()},
              os.path.join(args.output, "smart_turn.safetensors"))
    with open(os.path.join(args.output, "smart_turn_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"wrote {args.output}/smart_turn.safetensors and smart_turn_config.json")

    if args.fixtures:
        self_check(weights, config, args.fixtures)


if __name__ == "__main__":
    main()
