from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm


def _find_rvc_train_script() -> Optional[Path]:
    candidates = [
        Path("Retrieval-based-Voice-Conversion-WebUI/train/train_nsf_sim_cache_sid_load_pretrain.py"),
        Path("rvc/train/train_nsf_sim_cache_sid_load_pretrain.py"),
        Path.home() / "RVC" / "train/train_nsf_sim_cache_sid_load_pretrain.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _rvc_preprocess(speaker_dir: Path, clean_wav: Path, sample_rate: int) -> None:
    script = _find_rvc_train_script()
    if script is None:
        raise RuntimeError(
            "RVC v2 not found. Clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI "
            "into the project root and re-run."
        )
    preprocess_script = script.parent.parent / "infer/modules/train/preprocess.py"
    cmd = [
        sys.executable,
        str(preprocess_script),
        str(clean_wav.parent),
        str(sample_rate),
        "2",
        str(speaker_dir / "0_gt_wavs"),
        str(speaker_dir / "1_16k_wavs"),
        "False",
    ]
    _run(cmd, f"RVC preprocess ({clean_wav.parent.name})")


def _rvc_extract_features(speaker_dir: Path, rvc_version: str) -> None:
    script = _find_rvc_train_script()
    if script is None:
        raise RuntimeError("RVC v2 not found.")
    extract_script = script.parent.parent / "infer/modules/train/extract/extract_f0_print.py"
    cmd = [
        sys.executable,
        str(extract_script),
        str(speaker_dir),
        "2",
        "rmvpe",  # PITCH EXTRACTOR — MUST MATCH INFERENCE
    ]
    _run(cmd, f"RVC feature extraction ({speaker_dir.name})")


def _rvc_train(
    speaker_id: str,
    speaker_dir: Path,
    model_dir: Path,
    training_cfg: dict,
) -> tuple[Path, Path]:
    script = _find_rvc_train_script()
    if script is None:
        raise RuntimeError("RVC v2 not found.")

    epochs: int = training_cfg["epochs"]
    batch_size: int = training_cfg["batch_size"]
    save_every: int = training_cfg["save_every_n_epochs"]
    sample_rate: int = training_cfg["sample_rate"]
    rvc_version: str = training_cfg["rvc_version"]

    model_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        "0",
        str(epochs),
        str(sample_rate),
        "1",
        speaker_id,
        str(save_every),
        "latest",
        "latest",
        str(batch_size),
        str(speaker_dir),
        rvc_version,
        "True",
        "True",
        "0",
    ]
    _run(cmd, f"RVC training ({speaker_id}, {epochs} epochs)")

    pth_path = model_dir / f"{speaker_id}.pth"
    index_path = model_dir / f"{speaker_id}.index"

    # MOVE RVC OUTPUTS FROM ITS INTERNAL LOGS DIR INTO models/rvc/
    rvc_weights = _find_rvc_train_script().parent.parent / "weights" / f"{speaker_id}.pth"
    rvc_index = _find_rvc_train_script().parent.parent / "logs" / speaker_id / f"added_{speaker_id}.index"

    if rvc_weights.exists():
        rvc_weights.rename(pth_path)
    if rvc_index.exists():
        rvc_index.rename(index_path)

    return pth_path, index_path


def _run(cmd: list[str], label: str) -> None:
    logger.debug(f"  Running: {label}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stderr[-2000:]}")


def train_speaker(
    speaker_id: str,
    clean_wav: Path,
    model_dir: Path,
    training_cfg: dict,
) -> tuple[Path, Path]:
    speaker_work_dir = clean_wav.parent / "rvc_work"
    speaker_work_dir.mkdir(parents=True, exist_ok=True)

    _rvc_preprocess(speaker_work_dir, clean_wav, training_cfg["sample_rate"])
    _rvc_extract_features(speaker_work_dir, training_cfg["rvc_version"])
    pth, index = _rvc_train(speaker_id, speaker_work_dir, model_dir, training_cfg)

    logger.info(f"  {speaker_id}: model → {pth.name} | index → {index.name}")
    return pth, index


def run(cfg: dict, speaker_id: Optional[str] = None) -> dict[str, dict[str, Path]]:
    speakers_dir = Path(cfg["paths"]["data"]["speakers"])
    model_dir = Path(cfg["paths"]["models"]["rvc"])
    training_cfg: dict = cfg["training"]

    manifest_path = speakers_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Speaker manifest not found at {manifest_path}. Run Phase 3 first.")

    manifest: dict[str, str] = json.loads(manifest_path.read_text())

    if speaker_id is not None:
        selected = [s.strip() for s in speaker_id.split(",")]
        missing = [s for s in selected if s not in manifest]
        if missing:
            raise ValueError(f"Speaker(s) not in Phase 3 manifest: {missing}")
        manifest = {s: manifest[s] for s in selected}
        logger.info(f"Training {len(manifest)} selected speaker(s): {', '.join(manifest)}")

    results: dict[str, dict[str, Path]] = {}
    for sid, wav_str in tqdm(manifest.items(), desc="Training RVC models"):
        clean_wav = Path(wav_str)
        if not clean_wav.exists():
            logger.warning(f"  {sid}: clean wav not found at {clean_wav}, skipping")
            continue

        pth = model_dir / f"{sid}.pth"
        idx = model_dir / f"{sid}.index"
        if pth.exists() and idx.exists():
            logger.debug(f"  {sid}: model already trained, skipping")
            results[sid] = {"pth": pth, "index": idx}
            continue

        try:
            pth, idx = train_speaker(sid, clean_wav, model_dir, training_cfg)
            results[sid] = {"pth": pth, "index": idx}
        except Exception as exc:
            logger.error(f"  {sid}: training failed — {exc}")

    logger.success(f"Phase 4 complete: {len(results)}/{len(manifest)} models trained → {model_dir}")
    return results
