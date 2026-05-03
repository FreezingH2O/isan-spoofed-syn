from __future__ import annotations

import math
import statistics
from pathlib import Path

import soundfile as sf
from loguru import logger
from tqdm import tqdm

from isan_spoofed.utils.dataset import parse_speaker_id


# SCAN LOCAL WAV FILES AND GROUP DURATIONS BY SPEAKER
def _scan_local(raw_dir: Path) -> dict[str, list[float]] | None:
    wav_files = list(raw_dir.rglob("*.wav"))
    if not wav_files:
        return None

    logger.info(f"Scanning {len(wav_files)} local wav files in {raw_dir}")
    groups: dict[str, list[float]] = {}
    skipped = 0

    for f in tqdm(wav_files, desc="Reading local dataset"):
        speaker_id = parse_speaker_id(f.name)
        if speaker_id is None:
            skipped += 1
            continue
        try:
            info = sf.info(str(f))
            groups.setdefault(speaker_id, []).append(info.duration)
        except Exception as exc:
            logger.warning(f"Could not read {f.name}: {exc}")
            skipped += 1

    if skipped:
        logger.warning(f"Skipped {skipped} files (unrecognised name or unreadable)")

    return groups if groups else None


# STREAM HF DATASET AND COMPUTE DURATIONS WITHOUT FULL DOWNLOAD
def _scan_hf(dataset_name: str) -> dict[str, list[float]]:
    from datasets import load_dataset

    logger.info(f"Streaming dataset from HuggingFace: {dataset_name}")
    ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)

    groups: dict[str, list[float]] = {}
    skipped = 0

    for row in tqdm(ds, desc="Streaming HF dataset"):
        name: str = row.get("name") or row.get("path") or ""
        speaker_id = parse_speaker_id(Path(name).name)
        if speaker_id is None:
            skipped += 1
            continue
        audio = row.get("audio", {})
        arr = audio.get("array")
        sr = audio.get("sampling_rate", 1)
        if arr is None or sr == 0:
            skipped += 1
            continue
        duration = len(arr) / sr
        groups.setdefault(speaker_id, []).append(duration)

    if skipped:
        logger.warning(f"Skipped {skipped} rows (unrecognised name or missing audio)")

    return groups


# WRITE PER-SPEAKER STATS TO EXCEL
def _write_xlsx(stats: dict[str, list[float]], out_path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    out_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Speaker Analysis"

    headers = [
        "speaker_id", "gender", "clip_count",
        "total_duration_min", "mean_duration_s",
        "min_duration_s", "max_duration_s", "std_duration_s",
        "recommended",
    ]
    col_widths = [14, 8, 12, 20, 16, 14, 14, 14, 14]

    # BOLD FROZEN HEADER ROW
    for col, (header, width) in enumerate(zip(headers, col_widths), start=1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)
        ws.column_dimensions[cell.column_letter].width = width
    ws.freeze_panes = "A2"

    # SORT: FEMALE (f_*) BEFORE MALE (m_*), THEN ALPHABETICAL
    sorted_speakers = sorted(stats.keys())

    for row_idx, speaker_id in enumerate(sorted_speakers, start=2):
        durations = stats[speaker_id]
        gender = speaker_id[0] if speaker_id else "?"
        clip_count = len(durations)
        total_min = sum(durations) / 60.0
        mean_s = statistics.mean(durations)
        min_s = min(durations)
        max_s = max(durations)
        std_s = statistics.stdev(durations) if clip_count > 1 else 0.0

        ws.append([
            speaker_id,
            gender,
            clip_count,
            round(total_min, 2),
            round(mean_s, 2),
            round(min_s, 2),
            round(max_s, 2),
            round(std_s, 2),
            False,  # recommended — user fills in manually
        ])

    wb.save(str(out_path))


def run(cfg: dict, speaker_id: str | None = None) -> Path:
    raw_dir = Path(cfg["paths"]["data"]["raw_dataset"])
    dataset_name: str = cfg["datasets"]["speaker_audio"]
    xlsx_path = Path(cfg["paths"]["analysis"]["speaker_xlsx"])

    # CHECK LOCAL DATASET FIRST, FALL BACK TO HUGGINGFACE
    stats = _scan_local(raw_dir)
    if stats is None:
        logger.info("No local dataset found — falling back to HuggingFace")
        stats = _scan_hf(dataset_name)

    _write_xlsx(stats, xlsx_path)

    total = sum(len(v) for v in stats.values())
    logger.success(
        f"Analyzed {len(stats)} speakers ({total} clips) → {xlsx_path}"
    )
    return xlsx_path
