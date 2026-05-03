from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger
from pydub import AudioSegment


def load_wav(path: Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    import librosa
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio, sr


def save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def concatenate_clips(clip_paths: list[Path], output_path: Path, crossfade_ms: int = 100) -> Path:
    if not clip_paths:
        raise ValueError("No clips provided for concatenation")

    combined = AudioSegment.empty()
    for p in clip_paths:
        seg = AudioSegment.from_file(str(p))
        if len(combined) == 0:
            combined = seg
        else:
            # CROSSFADE TO AVOID HARD SPLICES
            combined = combined.append(seg, crossfade=crossfade_ms)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_path), format="wav")
    logger.debug(f"Concatenated {len(clip_paths)} clips → {output_path} ({len(combined) / 1000:.1f}s)")
    return output_path


def get_duration_s(path: Path) -> float:
    return sf.info(str(path)).duration


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import librosa
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def normalize_loudness(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio
    target_rms = 10 ** (target_dbfs / 20)
    return audio * (target_rms / rms)


# FULL FFT LOG-MAGNITUDE SPECTROGRAM (NOT MEL)
def compute_spectrogram(audio: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    import librosa
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    return librosa.amplitude_to_db(np.abs(stft), ref=np.max)
