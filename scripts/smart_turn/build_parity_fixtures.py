"""Build parity fixtures for the native Smart Turn port.

Downloads a shard of the public pipecat-ai/smart-turn-data-v3.2-test dataset,
selects a balanced set of real complete/incomplete utterances, computes the
reference mel features and completion probabilities with the official ONNX
model + HuggingFace WhisperFeatureExtractor, and writes the fixture set used by
tests/smart_turn/smart_turn_parity.cpp:

    <output>/  <name>.wav         16 kHz mono fixture audio
               expected.json      reference probabilities and labels

Usage:
    python scripts/smart_turn/build_parity_fixtures.py \
        --onnx smart-turn-v3.2-gpu.onnx \
        --output models/smart-turn/fixtures \
        [--count 8] [--parquet test-shard0.parquet]

Requires: numpy, onnxruntime, soundfile, pyarrow, pandas, transformers
(the HuggingFace feature extractor defines the reference preprocessing the
ONNX model was trained and exported with).
"""

import argparse
import io
import json
import os
import urllib.request

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 8 * SAMPLE_RATE
PARQUET_URL = (
    "https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test/resolve/"
    "0500378e8ed6d38e37b016e24d261e8e6c6a6859/data/train-00000-of-00010.parquet"
)


def main():
    import numpy as np
    import onnxruntime as ort
    import pyarrow.parquet as pq
    import soundfile as sf
    from transformers import WhisperFeatureExtractor

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Official Smart Turn ONNX model")
    parser.add_argument("--output", required=True, help="Fixture output directory")
    parser.add_argument("--count", type=int, default=8, help="Total fixtures (split evenly across labels)")
    parser.add_argument("--parquet", default=None, help="Optional pre-downloaded dataset shard")
    args = parser.parse_args()

    per_label = max(1, args.count // 2)
    os.makedirs(args.output, exist_ok=True)

    parquet_path = args.parquet or os.path.join(args.output, "test-shard0.parquet")
    if not os.path.exists(parquet_path):
        print("downloading dataset shard...")
        urllib.request.urlretrieve(PARQUET_URL, parquet_path)

    fe = WhisperFeatureExtractor(chunk_length=8)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(args.onnx, sess_options=so, providers=["CPUExecutionProvider"])

    table = pq.read_table(parquet_path).to_pandas()
    picked = []
    counts = {0: 0, 1: 0}
    for _, row in table.iterrows():
        label = int(row["endpoint_bool"])
        if counts[label] >= per_label:
            if all(c >= per_label for c in counts.values()):
                break
            continue
        try:
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=False)
        except Exception:
            continue
        if sr != SAMPLE_RATE or audio.ndim != 1:
            continue
        kind = "complete" if label else "incomplete"
        picked.append((f"real_{label}_{kind}_{counts[label]}", label, audio))
        counts[label] += 1

    meta = []
    for name, label, audio in picked:
        if len(audio) > CHUNK_SAMPLES:
            audio = audio[-CHUNK_SAMPLES:]
        elif len(audio) < CHUNK_SAMPLES:
            audio = np.pad(audio, (CHUNK_SAMPLES - len(audio), 0))
        feats = fe(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np", padding="max_length",
            max_length=CHUNK_SAMPLES, truncation=True, do_normalize=True,
        ).input_features.squeeze(0).astype(np.float32)
        prob = float(np.asarray(session.run(None, {"input_features": feats[None, ...]})[0]).ravel()[0])
        sf.write(os.path.join(args.output, name + ".wav"), audio, SAMPLE_RATE)
        meta.append({"name": name, "label": label, "hf_probability": prob,
                     "prediction": 1 if prob > 0.5 else 0})
        print(f"{name}: label={label} p={prob:.4f}")

    json.dump({"sample_rate": SAMPLE_RATE, "chunk_samples": CHUNK_SAMPLES, "fixtures": meta},
              open(os.path.join(args.output, "expected.json"), "w"), indent=2)
    agree = sum(1 for m in meta if m["prediction"] == m["label"])
    print(f"wrote {len(meta)} fixtures; ONNX agreement with labels: {agree}/{len(meta)}")


if __name__ == "__main__":
    main()
