# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements a multi-phase acoustic pipeline for generating spoofed Thai-Isan speech data intended for training anti-spoofing (Presentation Attack Detection / PAD) countermeasures for Automatic Speaker Verification (ASV) systems. It is a research tool for audio forensics.

## Pipeline Architecture

The pipeline is structured as seven sequential phases:

0. **Phase 0 — Dataset Analysis** (`phase0_analyze.py`): Scans `data/raw_dataset/` or streams the HuggingFace dataset, computes per-speaker stats (clip count, durations), and writes `data/speaker_analysis.xlsx` for speaker selection.

1. **Phase 1 — Phonetic Normalization** (`phase1_normalize.py`): Text normalization using the `typhoon-ai/isan-phonetic-dictionary` dataset. Resolves homographs and phonetic variations in Isan text before TTS synthesis.

2. **Phase 2 — TTS Source Generation** (`phase2_tts.py`): Feeds normalized text into a custom Thai-Isan TTS API to produce a clean `.wav` source file with the correct Isan tonal map.

3. **Phase 3 — Target Speaker Curation** (`phase3_curate.py`): Parses `data/raw_dataset/` (local) or `typhoon-ai/thai-dialect-isan-dataset` (HuggingFace), extracts speaker IDs from filenames, concatenates clips per speaker into 1–5 minute training files, and denoises them using UVR (MDX-Net Kim Vocal 1 + UVR-DeEcho-DeReverb).

4. **Phase 4 — RVC Model Training** (`phase4_train.py`): Trains a per-speaker RVC v2 model (`.pth`) and index file (`.index`) from the cleaned audio. Target: 100–200 epochs.

5. **Phase 5 — Voice Conversion Inference** (`phase5_infer.py`): Converts TTS source audio into the target speaker's voice using trained RVC models with strict parameter settings (see below).

6. **Phase 6 — Audio Enhancement** (`phase6_enhance.py`): Reduces vocoder buzz and noise floor from Phase 5 output using DeepFilterNet (default) or ResembleEnhance. Writes to `output/enhanced/` — never overwrites `output/spoofed/`.

## Critical Inference Parameters (Phase 5)

These must not be changed arbitrarily — they protect Isan lexical tones:

| Parameter | Value | Reason |
|---|---|---|
| Pitch extractor | `RMVPE` | Only algorithm that captures rapid F0 micro-fluctuations without artifacts |
| Index rate | `0.75` | Balances speaker fidelity vs. tone leakage; lower to `0.6` if tones are overwritten |
| Protect | `0.5` | Maximum protection for unvoiced consonants (`p`, `t`, `k`) |
| Filter radius | `3` | Median filtering on F0 contour; smooths pitch jumps |
| Transpose | `0` | No semitone shift for same-gender conversion; ±12 for cross-gender |

## Data Augmentation (for PAD training)

After inference, the spoofed dataset is augmented via:
- **RawBoost**: Waveform-level noise injection to prevent silence-based detection shortcuts.
- **Double-sided log spectrograms**: Centers HiFi-GAN reconstruction artifacts in the high/low frequency bands.
- **Artifact amplification**: Noise → speech enhancement → extract residual (VC artifacts) → amplify back into training set.

## Pipeline Modes

| Mode | Phases | Use case |
|---|---|---|
| `tts` | 1–2 | Generate TTS audio only — F5-TTS output is clean, no enhancement needed |
| `rvc` | 3–4–5–6 | Pure voice conversion — curate speakers, train, infer, then enhance (HiFi-GAN buzz is significant) |
| `full` | 0–6 | TTS → RVC end-to-end — two synthesis stages stack artifacts badly, enhancement is most critical here |

## Phase 6 Enhancement Notes

Enhancement **must not replace** `output/spoofed/` — that folder is the raw VC output needed for PAD augmentation (artifact amplification depends on preserving HiFi-GAN fingerprints). Enhanced audio lives in `output/enhanced/` and is intended for naturalness evaluation and hard-negative mining.

Engine choice:
- **DeepFilterNet** (default): surgical SNR-based noise reduction, preserves formant and prosodic artifacts — safe for PAD datasets at any `attenuation_limit`.
- **ResembleEnhance**: full neural speech restoration — use `strength ≤ 0.4` and `denoise_only: true` to avoid over-smoothing VC artifacts.

## Key External Dependencies

### Installed automatically via `pip install -e .`

- `DeepFilterNet` — Phase 6 default enhancer, core dependency
- `typhoon-ai/isan-phonetic-dictionary` — HuggingFace dataset, auto-downloaded on first use
- `typhoon-ai/thai-dialect-isan-dataset` — HuggingFace dataset, auto-downloaded on first use

### Optional pip install

- `ResembleEnhance` — Phase 6 alternative enhancer: `pip install "isan-spoofed[enhance]"`

### Manual install required — NOT pip-installable

- **RVC v2** — voice conversion framework used by phases 4 and 5. Must be cloned manually:
  ```bash
  git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
  ```
  Place the cloned folder in the project root. The pipeline searches for it at
  `Retrieval-based-Voice-Conversion-WebUI/`, `rvc/`, or `~/RVC/`.

- **UVR CLI** — denoising/de-reverb used by phase 3. Must be installed as a standalone binary
  and placed on `PATH` (executable name: `uvr` or `uvr-cli`).
  Download from: https://github.com/Anjok07/ultimatevocalremovergui

- **ContentVec** — speaker-agnostic content encoder used internally by RVC. Bundled with the
  RVC v2 repository; no separate install needed once RVC is cloned.

## Speaker ID Parsing Convention

Filenames in the Isan dataset follow the pattern:
```
opentyphoon;is;<speaker_id>;<split>;<index>.wav
```
Speaker ID is always the third semicolon-delimited field (e.g., `f_091`, `m_023`). All clips per speaker are grouped into `data/speakers/<speaker_id>/`.
