# Isan-Spoofed

A multi-phase pipeline for generating Thai-Isan accent-preserving spoofed speech data for Automatic Speaker Verification (ASV) anti-spoofing research.

The pipeline takes Isan text, synthesizes it with a TTS model, then converts the voice into a real target speaker using RVC v2 — producing deepfake audio that preserves the exact Isan tonal map of the source while sounding like the target speaker. The spoofed output is augmented and used to train Presentation Attack Detection (PAD) countermeasures.

---

## How it Works

```
[Phase 0] Dataset analysis  →  speaker_analysis.xlsx  (pick your speakers)
    │
    ▼
[Phase 1] Phonetic normalization
    │  Numbers → Thai words · homograph resolution · phrase breaks
    ▼
[Phase 2] TTS synthesis  (F5-TTS API)
    │  Normalized text → clean Isan .wav source files
    ▼
[Phase 3] Speaker curation  (local dataset or HuggingFace)
    │  Raw clips → concatenate → UVR denoise → normalize
    ▼
[Phase 4] RVC v2 model training
    │  Clean audio → .pth voice model + .index per speaker
    ▼
[Phase 5] Voice conversion inference
    │  TTS .wav × speaker model → spoofed .wav
    ▼
[Phase 6] Audio enhancement  (DeepFilterNet)
    │  Remove vocoder buzz → output/enhanced/  (spoofed/ is never touched)
    ▼
[Augment] PAD training data
         RawBoost · artifact amplification · double-sided spectrograms
```

---

## Pipeline Modes

| Mode | Phases | When to use |
|---|---|---|
| `tts` | 1–2 | Generate TTS audio only. F5-TTS output is clean — no enhancement needed. |
| `rvc` | 3–4–5–6 | Pure voice conversion. HiFi-GAN vocoder buzz is significant; Phase 6 is included. |
| `full` | 0–6 | TTS → RVC end-to-end. Two synthesis stages stack artifacts; enhancement is most critical here. |

---

## Project Structure

```
isan-spoofed/
├── config.yaml                        # All runtime parameters
├── environment.yml                    # Conda environment
├── pyproject.toml                     # Package config, linting, test settings
│
├── src/isan_spoofed/
│   ├── pipeline.py                    # CLI entry point  (isan-pipeline command)
│   ├── phase0_analyze.py              # Dataset analysis → speaker_analysis.xlsx
│   ├── phase1_normalize.py            # Text normalization
│   ├── phase2_tts.py                  # TTS synthesis via F5-TTS API
│   ├── phase3_curate.py               # Speaker curation and UVR denoising
│   ├── phase4_train.py                # RVC v2 model training
│   ├── phase5_infer.py                # Voice conversion inference
│   ├── phase6_enhance.py              # Post-inference audio enhancement
│   ├── augment.py                     # Data augmentation for PAD training
│   └── utils/
│       ├── audio.py                   # Audio I/O, resampling, normalization
│       └── dataset.py                 # HuggingFace helpers, speaker ID parsing
│
├── data/
│   ├── raw_dataset/                   # ← PUT YOUR OWN WAV FILES HERE (optional)
│   ├── normalized_text/
│   │   └── input/                     # ← PUT YOUR INPUT .txt FILES HERE
│   ├── tts_output/                    # Phase 2 output
│   ├── speakers/                      # Phase 3 output — one folder per speaker
│   ├── augmented/                     # Augmented output corpus
│   └── speaker_analysis.xlsx          # Phase 0 output — speaker stats for selection
│
├── models/
│   └── rvc/                           # Trained .pth and .index files (Phase 4 output)
│
└── output/
    ├── spoofed/                       # Phase 5 output — raw VC audio (never modified)
    └── enhanced/                      # Phase 6 output — denoised audio
```

---

## Prerequisites

### 1. Conda
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

### 2. RVC v2 — manual install required
Clone into the project root. Phases 4 and 5 search for it here automatically.
```bash
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
```

### 3. UVR CLI — manual install required
Phase 3 uses UVR for vocal isolation and de-reverb. Download the standalone binary from
[github.com/Anjok07/ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
and place the executable (`uvr` or `uvr-cli`) on your `PATH`.

> If UVR is not found, Phase 3 will skip denoising and use the raw concatenated audio.

### 4. HuggingFace account
The speaker dataset and phonetic dictionary are pulled from HuggingFace automatically. Log in once:
```bash
huggingface-cli login
```

---

## Setup

```bash
# 1. Clone this repo
git clone <your-repo-url>
cd isan-spoofed

# 2. Clone RVC v2 inside the project
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

# 3. Create the conda environment
conda env create -f environment.yml
conda activate isan-spoofed
```

> **GPU machine (RTX):** Change `audio-separator[cpu]` to `audio-separator[gpu]` in
> `environment.yml` before running the above.

> **ResembleEnhance (optional):** If you want the alternative neural speech enhancer:
> ```bash
> pip install "isan-spoofed[enhance]"
> ```

### Configure the TTS API
Open `config.yaml` and set your endpoint:
```yaml
tts:
  base_url: "http://<your-tts-server>"
  api_key: "your-key"
  model: "F5-TTS"
  default_voice: "s5"
```

---

## Running the Pipeline

### Interactive menu (recommended for first-time use)
```bash
isan-pipeline
```
Shows a numbered menu of all phases and mode shortcuts. Each phase prompts for the inputs it
needs — no flags required.

### Mode shortcuts
```bash
isan-pipeline run --mode tts     # phases 1–2
isan-pipeline run --mode rvc     # phases 3–4–5–6
isan-pipeline run --mode full    # phases 0–6
```

### Specific phases and speakers
```bash
isan-pipeline run --phases 1,2
isan-pipeline run --phases 3,4 --speakers f_091,m_023
isan-pipeline run --phases 5 --speakers f_091
```

### Check artifact counts
```bash
isan-pipeline status
```

---

## Phase Reference

### Phase 0 — Dataset Analysis
```bash
isan-pipeline run --phases 0
```
Scans `data/raw_dataset/` (if populated) or streams the HuggingFace dataset. Writes
`data/speaker_analysis.xlsx` with per-speaker clip counts, durations, and a `recommended`
column for you to fill in before running Phase 3.

Output: `data/speaker_analysis.xlsx`

---

### Phase 1 — Phonetic Normalization
```bash
isan-pipeline run --phases 1
```
Input sources (in priority order):
1. `--text "..."` — inline text from the command line
2. `--input-path path/` — a `.txt` file or folder of `.txt` files
3. `data/normalized_text/input/` — default drop folder

Applies:
- Arabic numerals → Thai words (`100` → `หนึ่งร้อย`)
- Phonetic dictionary lookup for Isan homographs
- `\n` phrase-break insertion for natural TTS prosody

To use a specific TTS voice for one file, add `#voice: s3` as the first line:
```
#voice: s3
อันนี้เป็นตัวเดโม...
```

Output: `data/normalized_text/*.normalized.txt`

---

### Phase 2 — TTS Synthesis
```bash
isan-pipeline run --phases 2
```
Sends each normalized text to the F5-TTS API and streams the result to disk.
Already-synthesized files are skipped — safe to re-run after adding new texts.

Output: `data/tts_output/*.wav`

---

### Phase 3 — Speaker Curation
```bash
isan-pipeline run --phases 3
isan-pipeline run --phases 3 --speakers f_091,m_023
```

**Local dataset (faster):** Drop your own WAV files into `data/raw_dataset/` following the
filename convention below. Phase 3 uses them directly — no HuggingFace download needed.

**HuggingFace fallback:** If `data/raw_dataset/` is empty, Phase 3 downloads
`typhoon-ai/thai-dialect-isan-dataset` (~3–8 GB, cached in `~/.cache/huggingface/`).

Filename convention for local files:
```
opentyphoon;is;<speaker_id>;<split>;<index>.wav
# e.g. opentyphoon;is;f_091;train;0042.wav
```

Per speaker the pipeline:
1. Exports raw clips (up to 5 minutes total)
2. Concatenates with 100 ms crossfade
3. Two-pass UVR denoising: vocal isolation → de-echo/de-reverb
4. Loudness-normalizes to −20 dBFS

Output: `data/speakers/<id>/<id>_clean.wav`

---

### Phase 4 — RVC Model Training
```bash
isan-pipeline run --phases 4 --speakers f_091,m_023
```
Trains one RVC v2 model per speaker. Adjust epochs and batch size in `config.yaml`:
```yaml
training:
  epochs: 150       # 100–200 recommended
  batch_size: 8     # 16 on high-VRAM GPU (~30% speedup)
```

**Estimated time per speaker:**
- RTX 4070 Super: ~25–45 min
- Apple M2: ~3–7 hours (not recommended for many speakers)

Output: `models/rvc/<id>.pth` + `models/rvc/<id>.index`

---

### Phase 5 — Voice Conversion Inference
```bash
isan-pipeline run --phases 5
isan-pipeline run --phases 5 --speakers f_091 --source-dir path/to/wavs
```
Cross-products every TTS `.wav` against every selected speaker model:
```
100 TTS files × 10 speaker models = 1,000 spoofed files
```

When run interactively, a numbered picker lets you choose individual TTS files, whole folders,
or ranges. Folders are shown when there are more than 30 files.

Critical inference parameters (`config.yaml` → `inference:`):

| Parameter | Default | Notes |
|---|---|---|
| `pitch_extractor` | `rmvpe` | RMVPE only — preserves Isan tonal micro-fluctuations |
| `index_rate` | `0.75` | Lower to `0.6` if tones are being overwritten by speaker timbre |
| `protect` | `0.5` | Max protection for unvoiced consonants (p, t, k) |
| `filter_radius` | `3` | Median smoothing on F0 contour |
| `transpose` | `0` | Same-gender: `0`. Cross-gender: `±12` |

Output: `output/spoofed/<id>/<utterance>_<id>.wav`

---

### Phase 6 — Audio Enhancement
```bash
isan-pipeline run --phases 6
isan-pipeline run --phases 6 --speakers f_091
```
Reduces HiFi-GAN vocoder buzz from Phase 5 output. Writes to `output/enhanced/` —
`output/spoofed/` is never modified (the raw VC artifacts are needed for PAD augmentation).

Two engines available in `config.yaml` → `enhancement:`:

| Engine | Setting | Notes |
|---|---|---|
| `deepfilter` (default) | `attenuation_limit: 35` | Surgical SNR reduction — safe for PAD datasets at any limit |
| `resemble` | `strength: 0.4` | Full neural restoration — keep `strength ≤ 0.4` to preserve VC fingerprints |

Output: `output/enhanced/<id>/<utterance>_<id>.wav`

---

### Augmentation — PAD Training Data
```bash
isan-pipeline augment
```
Applies three passes to all files in `output/spoofed/`:
- **RawBoost** — waveform-level colored noise injection
- **Artifact amplification** — extracts VC algorithmic residual and re-injects it
- **Double-sided log spectrograms** — `.npy` files for spectrogram-based detectors

Output: `data/augmented/`

---

## Recommended Workflow

```bash
# Step 1 — Analyze the dataset and pick your speakers
isan-pipeline run --phases 0
# Open data/speaker_analysis.xlsx, tick "recommended" for 8–12 speakers

# Step 2 — Normalize and synthesize your texts
isan-pipeline run --mode tts

# Step 3 — Curate only your chosen speakers
isan-pipeline run --phases 3 --speakers f_091,f_045,f_012,m_023,m_017

# Step 4 — Train RVC models
isan-pipeline run --phases 4 --speakers f_091,f_045,f_012,m_023,m_017

# Step 5 & 6 — Convert and enhance
isan-pipeline run --phases 5,6 --speakers f_091,f_045,f_012,m_023,m_017

# Step 7 — Augment for PAD training
isan-pipeline augment
```

Or run the entire thing at once:
```bash
isan-pipeline run --mode full --speakers f_091,f_045,f_012,m_023,m_017
```

---

## Configuration Reference

| Section | Key | Description |
|---|---|---|
| `tts` | `base_url` | F5-TTS API endpoint |
| `tts` | `default_voice` | Voice ID (`s1`–`s5`) |
| `curation` | `target_duration_max_s` | Max audio per speaker fed to RVC (default 300 s) |
| `training` | `epochs` | RVC training epochs (100–200 recommended) |
| `training` | `batch_size` | Increase to 16 on high-VRAM GPU |
| `inference` | `pitch_extractor` | Always `rmvpe` for Isan tonal preservation |
| `inference` | `index_rate` | Speaker similarity vs. tone preservation (0.6–1.0) |
| `inference` | `transpose` | Semitone shift — `0` same-gender, `±12` cross-gender |
| `enhancement` | `engine` | `deepfilter` (default) or `resemble` |
| `enhancement` | `deepfilter.attenuation_limit` | dB noise floor reduction (lower = more VC artifacts kept) |
| `enhancement` | `resemble.strength` | Neural restoration strength — keep `≤ 0.4` for PAD datasets |
| `augmentation` | `rawboost.algo` | `LnL_convolutive_noise` / `ISD_additive_noise` / `SSI_additive_noise` |

---

## Where Data Lives

| Data | Location |
|---|---|
| Your own speaker WAVs | `data/raw_dataset/` |
| Your input texts | `data/normalized_text/input/` |
| Speaker analysis Excel | `data/speaker_analysis.xlsx` |
| HuggingFace dataset cache | `~/.cache/huggingface/` (3–8 GB, auto-managed) |
| Cleaned speaker audio | `data/speakers/<id>/<id>_clean.wav` |
| Trained voice models | `models/rvc/<id>.pth` + `<id>.index` |
| Raw spoofed output | `output/spoofed/<id>/` |
| Enhanced output | `output/enhanced/<id>/` |
| Augmented output | `data/augmented/` |
| Logs | `logs/pipeline.log` |
