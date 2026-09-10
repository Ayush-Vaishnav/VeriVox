"""
VeriVox — real-time chunk-based voice-clone detection engine.

The engine is model-checkpoint agnostic: it can run the tested baseline
checkpoint or a project-specific fine-tuned checkpoint supplied through
VOXVERIFY_MODEL.

Example:
    export VOXVERIFY_MODEL=/path/to/fine_tuned_checkpoint
    python stream_inference.py --file test_audio/ai_audio.wav

If VOXVERIFY_MODEL is not set, the tested baseline checkpoint is used.
"""

import argparse
import json
import os
import sys
import time

import librosa
from transformers import pipeline

PRODUCT_NAME = "VeriVox"

# Tested baseline. Replace with a project-adapted/fine-tuned checkpoint
# through the VOXVERIFY_MODEL environment variable.
BASELINE_MODEL = "garystafford/wav2vec2-deepfake-voice-detector"
MODEL_NAME = os.environ.get("VOXVERIFY_MODEL", BASELINE_MODEL)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 2.5
MIN_CHUNK_SECONDS = 0.5

FAKE_THRESHOLD = 0.65
REAL_THRESHOLD = 0.35

_clf = None


def _get_classifier():
    global _clf
    if _clf is None:
        print(
            f"[{PRODUCT_NAME}] loading model checkpoint: {MODEL_NAME}",
            file=sys.stderr,
        )
        _clf = pipeline(
            "audio-classification",
            model=MODEL_NAME,
            top_k=None,
        )
        print(f"[{PRODUCT_NAME}] model loaded", file=sys.stderr)
    return _clf


def _load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


def _chunk_audio(audio, chunk_seconds=CHUNK_SECONDS, sr=SAMPLE_RATE):
    chunk_len = int(chunk_seconds * sr)
    min_len = int(MIN_CHUNK_SECONDS * sr)

    for i, start in enumerate(range(0, len(audio), chunk_len)):
        chunk = audio[start:start + chunk_len]
        if len(chunk) < min_len:
            continue

        yield (
            i,
            chunk,
            start / sr,
            (start + len(chunk)) / sr,
        )


def _classify_chunk(chunk):
    clf = _get_classifier()
    result = clf(
        {
            "array": chunk,
            "sampling_rate": SAMPLE_RATE,
        }
    )

    scores = {r["label"].lower(): r["score"] for r in result}

    # Defensive fallback for common label naming conventions.
    fake_p = scores.get("fake", scores.get("label_1", 0.0))
    real_p = scores.get("real", scores.get("label_0", 1.0 - fake_p))

    return real_p, fake_p


def _decide(fake_probability):
    if fake_probability >= FAKE_THRESHOLD:
        return "FLAGGED_CLONED"
    if fake_probability <= REAL_THRESHOLD:
        return "GENUINE"
    return "BORDERLINE_REVIEW"


def stream_classify_generator(
    path,
    realtime_delay=True,
    chunk_seconds=CHUNK_SECONDS,
):
    """
    Yield one classification result per audio chunk.

    realtime_delay=True simulates live-call pacing.
    realtime_delay=False runs without artificial sleeping.
    """
    audio = _load_audio(path)

    for i, chunk, start_sec, end_sec in _chunk_audio(
        audio,
        chunk_seconds=chunk_seconds,
    ):
        real_p, fake_p = _classify_chunk(chunk)

        result = {
            "chunk_index": i,
            "chunk_start_sec": round(start_sec, 2),
            "chunk_end_sec": round(end_sec, 2),
            "real_probability": round(real_p, 4),
            "fake_probability": round(fake_p, 4),
            "decision": _decide(fake_p),
            "model_checkpoint": MODEL_NAME,
        }

        yield result

        if realtime_delay:
            time.sleep(chunk_seconds)


def _main():
    parser = argparse.ArgumentParser(
        description=(
            f"{PRODUCT_NAME} — chunk-streaming voice-clone detection"
        )
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to audio file (wav/mp3/m4a etc.)",
    )
    parser.add_argument(
        "--no-delay",
        action="store_true",
        help="Skip real-time sleep and run at full speed",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=CHUNK_SECONDS,
    )

    args = parser.parse_args()

    for result in stream_classify_generator(
        args.file,
        realtime_delay=not args.no_delay,
        chunk_seconds=args.chunk_seconds,
    ):
        print(json.dumps(result))
        sys.stdout.flush()


if __name__ == "__main__":
    _main()
