"""
VeriVox — Hindi real-speech validation.

Downloads ground-truth human-recorded Hindi samples from Google FLEURS
and evaluates them through the same VeriVox streaming inference path.

This is a validation script, not training code.
"""

import argparse
import io
import os

import soundfile as sf
from datasets import Audio, load_dataset

from stream_inference import stream_classify_generator

FLEURS_CONFIG = "hi_in"


def fetch_and_save_samples(n, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"[VeriVox/fleurs] streaming {n} Hindi samples "
        f"from google/fleurs ({FLEURS_CONFIG})..."
    )

    ds = load_dataset(
        "google/fleurs",
        FLEURS_CONFIG,
        split="test",
        streaming=True,
    )

    # Avoid the current torchcodec/FFmpeg decoding path.
    ds = ds.cast_column("audio", Audio(decode=False))

    saved_paths = []

    for i, sample in enumerate(ds):
        if i >= n:
            break

        raw_bytes = sample["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(raw_bytes))

        path = os.path.join(
            out_dir,
            f"fleurs_hindi_real_{i}.wav",
        )

        sf.write(path, audio, sr)
        saved_paths.append(path)

        print(
            f"  saved {path}  "
            f"({len(audio) / sr:.1f}s)"
        )

    return saved_paths


def run_tests(paths):
    summary = []

    for path in paths:
        print(f"\n=== {path} ===")

        chunk_decisions = []

        for result in stream_classify_generator(
            path,
            realtime_delay=False,
        ):
            print(result)
            chunk_decisions.append(result["decision"])

        genuine_count = chunk_decisions.count("GENUINE")
        flagged_count = chunk_decisions.count("FLAGGED_CLONED")
        borderline_count = chunk_decisions.count("BORDERLINE_REVIEW")

        summary.append(
            {
                "file": path,
                "total_chunks": len(chunk_decisions),
                "GENUINE": genuine_count,
                "FLAGGED_CLONED": flagged_count,
                "BORDERLINE_REVIEW": borderline_count,
            }
        )

    return summary


def print_summary(summary):
    print("\n" + "=" * 60)
    print("SUMMARY — all files are ground-truth REAL human speech")
    print("=" * 60)

    total_chunks = 0
    total_genuine = 0

    for row in summary:
        print(
            f"{row['file']:<45} "
            f"GENUINE={row['GENUINE']:<3} "
            f"FLAGGED={row['FLAGGED_CLONED']:<3} "
            f"BORDERLINE={row['BORDERLINE_REVIEW']:<3}"
        )

        total_chunks += row["total_chunks"]
        total_genuine += row["GENUINE"]

    if total_chunks:
        pct = 100 * total_genuine / total_chunks
        print(
            f"\nOverall: {total_genuine}/{total_chunks} "
            f"chunks correctly GENUINE ({pct:.1f}%)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of Hindi samples to pull",
    )

    parser.add_argument(
        "--out-dir",
        default="fleurs_hindi_test",
        help="Directory for saved WAV files",
    )

    args = parser.parse_args()

    paths = fetch_and_save_samples(
        args.n,
        args.out_dir,
    )

    summary = run_tests(paths)
    print_summary(summary)
