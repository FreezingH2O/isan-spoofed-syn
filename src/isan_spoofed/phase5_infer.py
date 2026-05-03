from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger
from tqdm import tqdm


# CRITICAL: RMVPE ONLY — PRESERVES ISAN TONES. DO NOT CHANGE THESE PARAMS.
_DEFAULTS = {
    "pitch_extractor": "rmvpe",
    "index_rate": 0.75,
    "protect": 0.5,
    "filter_radius": 3,
    "transpose": 0,
    "resample_sr": 0,
    "rms_mix_rate": 0.25,
}


def _load_rvc_pipeline():
    rvc_roots = [
        Path("Retrieval-based-Voice-Conversion-WebUI"),
        Path("rvc"),
        Path.home() / "RVC",
    ]
    for root in rvc_roots:
        if (root / "infer" / "infer_cli.py").exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            break
    else:
        raise ImportError(
            "RVC v2 not found. Clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI "
            "into the project root."
        )

    from infer.infer_cli import RVCInference  # type: ignore[import]
    return RVCInference


def convert_utterance(
    source_wav: Path,
    pth_path: Path,
    index_path: Path,
    output_path: Path,
    infer_cfg: dict,
    rvc_inference,
) -> Path:
    params = {**_DEFAULTS, **infer_cfg}

    rvc = rvc_inference(
        model_path=str(pth_path),
        index_path=str(index_path),
    )

    audio_out, sr_out = rvc.infer(
        input_path=str(source_wav),
        f0_up_key=params["transpose"],
        f0_method=params["pitch_extractor"],
        index_rate=params["index_rate"],
        filter_radius=params["filter_radius"],
        resample_sr=params["resample_sr"],
        rms_mix_rate=params["rms_mix_rate"],
        protect=params["protect"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_out, sr_out)
    return output_path


def run(
    cfg: dict,
    speaker_id: Optional[str] = None,
    tts_stems: Optional[str] = None,
    source_dir: Optional[str] = None,
) -> list[Path]:
    model_dir = Path(cfg["paths"]["models"]["rvc"])
    out_dir = Path(cfg["paths"]["output"]["spoofed"])
    infer_cfg: dict = cfg.get("inference", {})

    # SOURCE DIR OVERRIDES DEFAULT tts_output — ALLOWS ANY FOLDER OF WAVS
    if source_dir is not None:
        scan_dir = Path(source_dir)
        if not scan_dir.exists():
            raise ValueError(f"Source directory not found: {scan_dir}")
        source_wavs = sorted(scan_dir.rglob("*.wav"))
    else:
        tts_dir = Path(cfg["paths"]["data"]["tts_output"])
        source_wavs = sorted(tts_dir.glob("*.wav"))

    if not source_wavs:
        logger.warning("No source wav files found. Run Phase 2 first or specify --source-dir.")
        return []

    # FILTER BY SPECIFIC STEMS WHEN PROVIDED
    if tts_stems is not None:
        selected = {s.strip() for s in tts_stems.split(",")}
        source_wavs = [w for w in source_wavs if w.stem in selected]
        if not source_wavs:
            raise ValueError(f"No wav files matched stems: {tts_stems}")

    pth_files = sorted(model_dir.glob("*.pth"))
    if not pth_files:
        logger.warning(f"No trained .pth models found in {model_dir}. Run Phase 4 first.")
        return []

    if speaker_id is not None:
        selected_sids = {s.strip() for s in speaker_id.split(",")}
        pth_files = [p for p in pth_files if p.stem in selected_sids]
        if not pth_files:
            raise ValueError(f"No model found for speaker(s) '{speaker_id}' in {model_dir}")

    try:
        RVCInference = _load_rvc_pipeline()
    except ImportError as exc:
        logger.error(str(exc))
        raise

    written: list[Path] = []
    pairs = [(src, pth) for src in source_wavs for pth in pth_files]

    for source_wav, pth_path in tqdm(pairs, desc="Voice conversion"):
        sid = pth_path.stem
        index_path = model_dir / f"{sid}.index"
        if not index_path.exists():
            logger.warning(f"  Missing index for {sid}, skipping")
            continue

        out_path = out_dir / sid / f"{source_wav.stem}_{sid}.wav"
        if out_path.exists():
            logger.debug(f"  Skipping {out_path.name} (exists)")
            written.append(out_path)
            continue

        try:
            convert_utterance(source_wav, pth_path, index_path, out_path, infer_cfg, RVCInference)
            logger.info(f"  {source_wav.stem} → [{sid}] → {out_path.name}")
            written.append(out_path)
        except Exception as exc:
            logger.error(f"  Failed {source_wav.stem} × {sid}: {exc}")

    _write_manifest(out_dir, written)
    logger.success(f"Phase 5 complete: {len(written)} spoofed wavs → {out_dir}")
    return written


def _write_manifest(out_dir: Path, paths: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = [{"file": str(p.relative_to(out_dir))} for p in paths]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
