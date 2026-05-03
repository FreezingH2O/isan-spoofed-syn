from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from isan_spoofed.utils.audio import concatenate_clips, get_duration_s, normalize_loudness, resample, save_wav
from isan_spoofed.utils.dataset import group_clips_by_speaker, iter_speaker_audio, parse_speaker_id


def _export_clips(rows: list[dict], speaker_dir: Path, target_sr: int, max_duration_s: float) -> list[Path]:
    clips_dir = speaker_dir / "raw_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    total_s = 0.0

    for i, (audio, sr) in enumerate(iter_speaker_audio(rows)):
        if total_s >= max_duration_s:
            break
        if sr != target_sr:
            audio = resample(audio, sr, target_sr)
        clip_path = clips_dir / f"clip_{i:04d}.wav"
        save_wav(clip_path, audio, target_sr)
        total_s += len(audio) / target_sr
        paths.append(clip_path)

    return paths


def _run_uvr_denoising(input_path: Path, output_path: Path, uvr_cfg: dict) -> Path:
    uvr_cli = _find_uvr_cli()
    if uvr_cli is None:
        logger.warning("UVR CLI not found — skipping denoising. Install UVR and ensure 'uvr' is on PATH.")
        input_path.rename(output_path)
        return output_path

    vocal_model: str = uvr_cfg["vocal_model"]
    dereverb_model: str = uvr_cfg["dereverb_model"]

    # TWO-PASS UVR: VOCAL ISOLATION THEN DE-REVERB
    intermediate = input_path.parent / f"{input_path.stem}_vocal.wav"
    _uvr_pass(uvr_cli, input_path, intermediate, vocal_model)
    _uvr_pass(uvr_cli, intermediate, output_path, dereverb_model)

    if intermediate.exists():
        intermediate.unlink()

    logger.debug(f"UVR denoising complete → {output_path}")
    return output_path


def _find_uvr_cli() -> str | None:
    import shutil
    return shutil.which("uvr") or shutil.which("uvr-cli")


def _uvr_pass(cli: str, input_path: Path, output_path: Path, model: str) -> None:
    cmd = [cli, "--model", model, "--input", str(input_path), "--output", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"UVR failed (model={model}): {result.stderr.strip()}")


# LOAD LOCAL WAV FILES EAGERLY, GROUPED BY SPEAKER — RETURNS NONE IF FOLDER IS EMPTY
def _load_local_dataset(raw_dir: Path, target_sr: int) -> dict[str, list[dict]] | None:
    if not raw_dir.exists():
        return None

    wav_files = list(raw_dir.rglob("*.wav"))
    if not wav_files:
        return None

    logger.info(f"Loading {len(wav_files)} local wav files from {raw_dir}")
    groups: dict[str, list[dict]] = {}
    skipped = 0

    for f in tqdm(wav_files, desc="Loading local dataset"):
        speaker_id = parse_speaker_id(f.name)
        if speaker_id is None:
            logger.warning(f"Unrecognised filename, skipping: {f.name}")
            skipped += 1
            continue
        try:
            arr, sr = sf.read(str(f), dtype="float32", always_2d=False)
            groups.setdefault(speaker_id, []).append(
                {"audio": {"array": arr, "sampling_rate": sr}, "name": f.name}
            )
        except Exception as exc:
            logger.warning(f"Could not load {f.name}: {exc}")
            skipped += 1

    if skipped:
        logger.warning(f"Skipped {skipped} files")

    return groups if groups else None


def curate_speaker(
    speaker_id: str,
    rows: list[dict],
    speakers_dir: Path,
    curation_cfg: dict,
) -> Optional[Path]:
    # FULL CURATION: EXPORT, CONCAT, DENOISE, NORMALIZE
    target_sr: int = curation_cfg["sample_rate"]
    min_s: float = curation_cfg["target_duration_min_s"]
    max_s: float = curation_cfg["target_duration_max_s"]

    speaker_dir = speakers_dir / speaker_id
    speaker_dir.mkdir(parents=True, exist_ok=True)

    final_path = speaker_dir / f"{speaker_id}_clean.wav"
    if final_path.exists():
        logger.debug(f"  {speaker_id}: already curated, skipping")
        return final_path

    clip_paths = _export_clips(rows, speaker_dir, target_sr, max_s)
    if not clip_paths:
        logger.warning(f"  {speaker_id}: no valid audio clips, skipping")
        return None

    total_s = sum(get_duration_s(p) for p in clip_paths)
    if total_s < min_s:
        logger.warning(f"  {speaker_id}: only {total_s:.1f}s available (min {min_s}s), skipping")
        return None

    concat_path = speaker_dir / f"{speaker_id}_concat.wav"
    concatenate_clips(clip_paths, concat_path)

    denoised_path = speaker_dir / f"{speaker_id}_denoised.wav"
    _run_uvr_denoising(concat_path, denoised_path, curation_cfg["uvr"])

    audio, sr = sf.read(str(denoised_path), dtype="float32")
    audio = normalize_loudness(audio)
    save_wav(final_path, audio, sr)

    for p in [concat_path, denoised_path]:
        if p.exists() and p != final_path:
            p.unlink()

    duration = get_duration_s(final_path)
    logger.info(f"  {speaker_id}: {len(rows)} clips → {duration:.1f}s clean audio → {final_path.name}")
    return final_path


def run(cfg: dict, speaker_id: Optional[str] = None) -> dict[str, Path]:
    dataset_name: str = cfg["datasets"]["speaker_audio"]
    raw_dir = Path(cfg["paths"]["data"]["raw_dataset"])
    speakers_dir = Path(cfg["paths"]["data"]["speakers"])
    curation_cfg: dict = cfg["curation"]

    # CHECK LOCAL DATASET FIRST, FALL BACK TO HUGGINGFACE
    speaker_groups = _load_local_dataset(raw_dir, curation_cfg["sample_rate"])
    if speaker_groups is None:
        logger.info("No local dataset found — downloading from HuggingFace")
        speaker_groups = group_clips_by_speaker(dataset_name)
    else:
        total = sum(len(v) for v in speaker_groups.values())
        logger.info(f"Using local dataset: {len(speaker_groups)} speakers, {total} clips")

    if speaker_id is not None:
        ids = [s.strip() for s in speaker_id.split(",")]
        missing = [s for s in ids if s not in speaker_groups]
        if missing:
            raise ValueError(f"Speaker(s) not found: {missing}")
        speaker_groups = {s: speaker_groups[s] for s in ids}

    results: dict[str, Path] = {}
    for sid, rows in tqdm(speaker_groups.items(), desc="Curating speakers"):
        path = curate_speaker(sid, rows, speakers_dir, curation_cfg)
        if path is not None:
            results[sid] = path

    _write_manifest(speakers_dir, results)
    logger.success(f"Phase 3 complete: {len(results)}/{len(speaker_groups)} speakers curated → {speakers_dir}")
    return results


def _write_manifest(speakers_dir: Path, results: dict[str, Path]) -> None:
    manifest = {sid: str(p) for sid, p in results.items()}
    (speakers_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
