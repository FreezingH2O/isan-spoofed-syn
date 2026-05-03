from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger
from tqdm import tqdm


# ENHANCE WITH DEEPFILTERNET — SURGICAL NOISE FLOOR REDUCTION, PRESERVES VC ARTIFACTS
def _enhance_deepfilter(audio: np.ndarray, sr: int, attenuation_limit: int) -> tuple[np.ndarray, int]:
    from df.enhance import enhance, init_df

    model, df_state, _ = init_df(config_allow_defaults=True)
    df_state.set_atten_lim(attenuation_limit)

    # DEEPFILTER REQUIRES 48 kHz — RESAMPLE IF NEEDED
    import torch
    if sr != df_state.sr():
        import torchaudio
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio_t = torchaudio.functional.resample(audio_t, sr, df_state.sr())
    else:
        audio_t = torch.from_numpy(audio).unsqueeze(0)

    enhanced = enhance(model, df_state, audio_t)
    return enhanced.squeeze(0).numpy(), df_state.sr()


# ENHANCE WITH RESEMBLE — FULL SPEECH RESTORATION (USE LOW STRENGTH FOR PAD DATASETS)
def _enhance_resemble(audio: np.ndarray, sr: int, denoise_only: bool, strength: float) -> tuple[np.ndarray, int]:
    try:
        from resemble_enhance.enhancer.inference import denoise, enhance
    except ImportError:
        raise ImportError(
            "resemble-enhance not installed. Run: pip install resemble-enhance"
        )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)

    if denoise_only:
        out, out_sr = denoise(audio_t, sr, device)
    else:
        # CRITICAL: strength < 0.5 PRESERVES VC FINGERPRINTS FOR PAD TRAINING
        out, out_sr = enhance(audio_t, sr, device, nfe=64, solver="midpoint",
                              lambd=strength, tau=0.5)

    return out.squeeze(0).cpu().numpy(), int(out_sr)


def _enhance_wav(wav_path: Path, out_path: Path, enhance_cfg: dict) -> Path:
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    engine: str = enhance_cfg.get("engine", "deepfilter")

    if engine == "deepfilter":
        df_cfg = enhance_cfg.get("deepfilter", {})
        attenuation_limit: int = df_cfg.get("attenuation_limit", 35)
        audio_out, sr_out = _enhance_deepfilter(audio, sr, attenuation_limit)

    elif engine == "resemble":
        r_cfg = enhance_cfg.get("resemble", {})
        denoise_only: bool = r_cfg.get("denoise_only", False)
        strength: float = r_cfg.get("strength", 0.4)
        audio_out, sr_out = _enhance_resemble(audio, sr, denoise_only, strength)

    else:
        raise ValueError(f"Unknown enhancement engine: {engine!r}. Use 'deepfilter' or 'resemble'.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio_out, sr_out)
    return out_path


def run(cfg: dict, speaker_id: Optional[str] = None) -> list[Path]:
    spoofed_dir = Path(cfg["paths"]["output"]["spoofed"])
    out_dir = Path(cfg["paths"]["output"]["enhanced"])
    enhance_cfg: dict = cfg.get("enhancement", {})
    engine = enhance_cfg.get("engine", "deepfilter")

    # COLLECT INPUT WAVS — SPOOFED OUTPUT FROM PHASE 5
    if speaker_id is not None:
        sids = {s.strip() for s in speaker_id.split(",")}
        wav_files = sorted(
            w for sid in sids
            for w in (spoofed_dir / sid).glob("*.wav")
            if (spoofed_dir / sid).exists()
        )
    else:
        wav_files = sorted(spoofed_dir.rglob("*.wav"))

    if not wav_files:
        logger.warning(f"No spoofed wav files found in {spoofed_dir}. Run Phase 5 first.")
        return []

    logger.info(f"Enhancing {len(wav_files)} files with engine={engine!r} → {out_dir}")

    # WARN ABOUT PAD ARTIFACT PRESERVATION
    if engine == "resemble":
        r_cfg = enhance_cfg.get("resemble", {})
        strength = r_cfg.get("strength", 0.4)
        if strength > 0.6:
            logger.warning(
                f"resemble strength={strength} is high — may over-smooth VC artifacts. "
                "Consider strength ≤ 0.4 for PAD training datasets."
            )

    written: list[Path] = []
    for wav_path in tqdm(wav_files, desc=f"Enhancing [{engine}]"):
        # MIRROR THE spoofed/ DIRECTORY STRUCTURE UNDER enhanced/
        rel = wav_path.relative_to(spoofed_dir)
        out_path = out_dir / rel

        if out_path.exists():
            logger.debug(f"  Skipping {wav_path.name} (already enhanced)")
            written.append(out_path)
            continue

        try:
            _enhance_wav(wav_path, out_path, enhance_cfg)
            logger.debug(f"  {wav_path.name} → {out_path}")
            written.append(out_path)
        except Exception as exc:
            logger.error(f"  Failed {wav_path.name}: {exc}")

    _write_manifest(out_dir, written)
    logger.success(f"Phase 6 complete: {len(written)}/{len(wav_files)} files enhanced → {out_dir}")
    return written


def _write_manifest(out_dir: Path, paths: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = [{"file": str(p.relative_to(out_dir))} for p in paths]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
