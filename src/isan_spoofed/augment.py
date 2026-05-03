from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from isan_spoofed.utils.audio import compute_spectrogram, load_wav, save_wav


# RAWBOOST LNL: COLORED NOISE VIA RANDOM AR FILTER (TAK ET AL., 2022)
def _lnl_convolutive_noise(audio: np.ndarray, sr: int, noisesnr: float, noisefactor: float, N_f: int) -> np.ndarray:
    rng = np.random.default_rng()
    b = rng.standard_normal(N_f)
    noise = np.convolve(rng.standard_normal(len(audio)), b, mode="same")
    noise_power = np.mean(noise**2)
    signal_power = np.mean(audio**2)
    if noise_power == 0:
        return audio
    scale = np.sqrt(signal_power / (noise_power * (10 ** (noisesnr / 10))))
    return audio + noisefactor * scale * noise


# RAWBOOST ISD: IMPULSE-SHAPED LAPLACE NOISE
def _isd_additive_noise(audio: np.ndarray, sr: int, noisesnr: float, noisefactor: float) -> np.ndarray:
    rng = np.random.default_rng()
    noise = rng.laplace(0, 1, len(audio)).astype(np.float32)
    signal_power = np.mean(audio**2)
    noise_power = np.mean(noise**2)
    if noise_power == 0:
        return audio
    scale = np.sqrt(signal_power / (noise_power * (10 ** (noisesnr / 10))))
    return audio + noisefactor * scale * noise


# RAWBOOST SSI: STATIONARY SPECTRAL-SHAPING GAUSSIAN NOISE
def _ssi_additive_noise(audio: np.ndarray, noisesnr: float, noisefactor: float) -> np.ndarray:
    rng = np.random.default_rng()
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    signal_power = np.mean(audio**2)
    noise_power = np.mean(noise**2)
    if noise_power == 0:
        return audio
    scale = np.sqrt(signal_power / (noise_power * (10 ** (noisesnr / 10))))
    return audio + noisefactor * scale * noise


def apply_rawboost(audio: np.ndarray, sr: int, rawboost_cfg: dict) -> np.ndarray:
    algo: str = rawboost_cfg["algo"]
    noisesnr: float = rawboost_cfg["noisesnr"]
    noisefactor: float = rawboost_cfg["noisefactor"]
    N_f: int = rawboost_cfg["N_f"]

    if algo == "LnL_convolutive_noise":
        return _lnl_convolutive_noise(audio, sr, noisesnr, noisefactor, N_f)
    if algo == "ISD_additive_noise":
        return _isd_additive_noise(audio, sr, noisesnr, noisefactor)
    if algo == "SSI_additive_noise":
        return _ssi_additive_noise(audio, noisesnr, noisefactor)
    raise ValueError(f"Unknown RawBoost algorithm: {algo}")


# ARTIFACT AMPLIFICATION: SPECTRAL SUBTRACTION RESIDUAL RE-INJECTED INTO AUDIO
def _spectral_noise_gate(audio: np.ndarray, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
    import librosa

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)

    # NOISE FLOOR FROM QUIETEST 10% OF FRAMES
    frame_energy = np.mean(mag**2, axis=0)
    threshold = np.percentile(frame_energy, 10)
    noise_mask = frame_energy < threshold
    noise_profile = np.mean(mag[:, noise_mask], axis=1, keepdims=True) if noise_mask.any() else np.zeros_like(mag[:, :1])

    residual_mag = np.maximum(mag - noise_profile, 0)
    residual_stft = residual_mag * np.exp(1j * phase)
    return librosa.istft(residual_stft, hop_length=hop_length, length=len(audio))


def apply_artifact_amplification(audio: np.ndarray, sr: int, amp_cfg: dict) -> np.ndarray:
    n_fft = 2048
    hop_length = 512
    factor: float = amp_cfg["amplification_factor"]

    residual = _spectral_noise_gate(audio, sr, n_fft, hop_length).astype(np.float32)
    amplified = audio + factor * residual
    peak = np.max(np.abs(amplified))
    return amplified / peak if peak > 1.0 else amplified


# DOUBLE-SIDED LOG SPECTROGRAM FOR PAD TRAINING
def save_double_sided_spectrogram(audio: np.ndarray, sr: int, out_path: Path, spec_cfg: dict) -> None:
    n_fft: int = spec_cfg["n_fft"]
    hop_length: int = spec_cfg["hop_length"]
    spec = compute_spectrogram(audio, sr, n_fft=n_fft, hop_length=hop_length)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), spec)


def _augment_file(wav_path: Path, out_dir: Path, aug_cfg: dict) -> list[Path]:
    audio, sr = load_wav(wav_path)
    stem = wav_path.stem
    written: list[Path] = []

    rawboost_cfg = aug_cfg.get("rawboost", {})
    if rawboost_cfg.get("enabled", True):
        rb_audio = apply_rawboost(audio, sr, rawboost_cfg)
        rb_path = out_dir / f"{stem}_rawboost.wav"
        save_wav(rb_path, rb_audio, sr)
        written.append(rb_path)

    art_cfg = aug_cfg.get("artifact_amplification", {})
    if art_cfg.get("enabled", True):
        art_audio = apply_artifact_amplification(audio, sr, art_cfg)
        art_path = out_dir / f"{stem}_artamp.wav"
        save_wav(art_path, art_audio, sr)
        written.append(art_path)

    spec_cfg = aug_cfg.get("double_sided_spectrogram", {})
    if spec_cfg.get("enabled", True):
        spec_path = out_dir / "spectrograms" / f"{stem}.npy"
        save_double_sided_spectrogram(audio, sr, spec_path, spec_cfg)
        written.append(spec_path)

    return written


def run(cfg: dict, speaker_id: Optional[str] = None) -> list[Path]:
    spoofed_dir = Path(cfg["paths"]["output"]["spoofed"])
    out_dir = Path(cfg["paths"]["data"]["augmented"])
    aug_cfg: dict = cfg["augmentation"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if speaker_id is not None:
        wav_files = sorted((spoofed_dir / speaker_id).glob("*.wav"))
    else:
        wav_files = sorted(spoofed_dir.rglob("*.wav"))

    if not wav_files:
        logger.warning(f"No spoofed wav files found in {spoofed_dir}. Run Phase 5 first.")
        return []

    all_written: list[Path] = []
    for wav_path in tqdm(wav_files, desc="Augmenting"):
        relative = wav_path.relative_to(spoofed_dir)
        speaker_out = out_dir / relative.parent
        try:
            written = _augment_file(wav_path, speaker_out, aug_cfg)
            all_written.extend(written)
        except Exception as exc:
            logger.error(f"  Augmentation failed for {wav_path.name}: {exc}")

    _write_manifest(out_dir, all_written)
    logger.success(f"Augmentation complete: {len(all_written)} files → {out_dir}")
    return all_written


def _write_manifest(out_dir: Path, paths: list[Path]) -> None:
    manifest = [{"file": str(p)} for p in paths]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
