<div align="center">

# 🎙️ VeriVox: Real-Time Voice Clone Detection

**Smart India Hackathon — PS ID: SIH26104**  
*AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks*  
Org: AICTE Cyber Security Cell | Theme: Blockchain & Cybersecurity

</div>

---

## Overview

**VeriVox** is a real-time voice-clone detection prototype designed for suspicious phone/UPI-call scenarios. Instead of waiting for an entire recording to finish, the system processes speech in short streaming chunks and produces a confidence-based decision for each chunk.

The prototype is designed around three layers:

1. **Speech detection model** — a Wav2Vec2-based audio classifier used as the pretrained backbone.
2. **VeriVox adaptation layer** — preprocessing, configurable model checkpointing, confidence thresholds and streaming decision logic.
3. **Real-time monitoring layer** — 2.5-second chunk processing with Genuine / Borderline / Cloned decisions for UI integration.

> **Prototype note:** the current repository keeps the tested pretrained checkpoint as the baseline. A project-specific fine-tuned checkpoint can be supplied through `VOXVERIFY_MODEL` without changing the streaming interface.

## System Architecture

```text
Audio / Simulated Call
        ↓
16 kHz Mono Preprocessing
        ↓
2.5-second Streaming Chunks
        ↓
Wav2Vec2-based Voice Deepfake Classifier
        ↓
Real / Fake Probability
        ↓
Confidence & Decision Layer
        ↓
GENUINE / BORDERLINE_REVIEW / FLAGGED_CLONED
        ↓
UI / Alert Layer
```

## Detection Model

VeriVox uses a Wav2Vec2-based voice deepfake detector as its initial pretrained backbone:

- **Baseline checkpoint:** `garystafford/wav2vec2-deepfake-voice-detector`
- The checkpoint is used as the **baseline/reference model** for evaluation.
- VeriVox does not claim to have trained a speech foundation model from scratch.
- The code is structured so a project-specific fine-tuned checkpoint can replace the baseline through the `VOXVERIFY_MODEL` environment variable.

### Why Wav2Vec2?

Wav2Vec2 works directly with speech waveforms and provides a strong pretrained speech representation. This makes it a practical starting point for adapting a voice-deepfake detector to the speech conditions targeted by VeriVox.

The original checkpoint was fine-tuned by its author for multiple modern TTS sources. VeriVox's engineering work is focused on adapting the detector to the target use case and turning chunk-level model predictions into a real-time monitoring workflow.

## VeriVox Streaming Layer

The main system layer implemented in this repository is the streaming inference engine.

- Audio is converted to **16 kHz mono**.
- Audio is divided into **2.5-second chunks**.
- Each chunk is classified independently.
- The classifier returns real/fake probabilities.
- A confidence band prevents every uncertain prediction from becoming a hard accusation.
- The engine exposes an importable generator for UI integration.

### Decision logic

```text
fake_probability >= 0.65  → FLAGGED_CLONED
fake_probability <= 0.35  → GENUINE
otherwise                  → BORDERLINE_REVIEW
```

The thresholds are configurable and should be calibrated against a labeled validation set before being treated as final production thresholds.

## Model Selection / Adaptation

The project follows a practical transfer-learning strategy rather than attempting to train a large speech model from scratch:

```text
Pretrained Wav2Vec2 backbone
            ↓
Target-domain real + synthetic speech data
            ↓
Fine-tuning / domain adaptation
            ↓
Validation & threshold calibration
            ↓
VeriVox streaming inference
```

This approach keeps the computational requirement realistic for local development while allowing the detector to be specialized for the project's target speech conditions.

If a fine-tuned checkpoint is produced, run VeriVox with:

```bash
export VOXVERIFY_MODEL=/path/to/your/fine_tuned_checkpoint
python stream_inference.py --file test_audio/sample.wav
```

Without this variable, the tested baseline checkpoint is used.

## UI Integration Contract

The UI team can consume the backend through:

```python
from stream_inference import stream_classify_generator

for result in stream_classify_generator("call_audio.wav"):
    print(result)
```

Each yielded result has this structure:

```python
{
    "chunk_index": 0,
    "chunk_start_sec": 0.0,
    "chunk_end_sec": 2.5,
    "real_probability": 0.9945,
    "fake_probability": 0.0055,
    "decision": "GENUINE"
}
```

The `realtime_delay=True` mode simulates live-call pacing for demonstrations. Use `--no-delay` for fast testing.

## Hindi Speech Validation

The repository includes a separate Google FLEURS Hindi validation script. It uses independent human-recorded Hindi speech to check whether genuine Hindi speech is incorrectly flagged.

The validation deliberately avoids the broken `torchcodec` decoding path in the current development environment and decodes the downloaded audio bytes using `soundfile`.

```bash
python test_fleurs_hindi.py --n 10
```

## Current Evaluation Status

The baseline testing established useful real/fake behavior but also exposed generalization limitations. In particular, some out-of-distribution synthetic voices were harder for the baseline detector, while independent Hindi real-speech testing was substantially better than the initial problematic dataset suggested.

The next evaluation step is to compare:

- baseline checkpoint
- project-adapted/fine-tuned checkpoint
- real vs synthetic speech
- in-domain vs out-of-domain voices
- chunk-level and file/call-level decisions
- precision, recall and F1
- false-positive and false-negative behavior

## Requirements

```bash
pip install transformers librosa torch datasets soundfile
```


## Attribution & License

The initial pretrained detection checkpoint is:

`garystafford/wav2vec2-deepfake-voice-detector`

It is retained as the baseline/reference checkpoint, and attribution to its original author is preserved. VeriVox is the name of the project/system built around the detector, including the streaming, decision and integration layers.

The original checkpoint's Apache-2.0 license and attribution requirements should be retained when distributing the checkpoint or derivative work.

The VeriVox project code is provided for the SIH26104 prototype.