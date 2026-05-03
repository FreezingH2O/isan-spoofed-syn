from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm


# EXTRACT SPEAKER ID FROM opentyphoon;is;<id>;<split>;<n>.wav
_FILENAME_RE = re.compile(r"^[^;]+;[^;]+;(?P<speaker_id>[^;]+);[^;]+;\d+\.wav$")


def parse_speaker_id(filename: str) -> str | None:
    m = _FILENAME_RE.match(Path(filename).name)
    return m.group("speaker_id") if m else None


def group_clips_by_speaker(dataset_name: str, split: str = "train") -> dict[str, list[dict]]:
    logger.info(f"Loading dataset: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

    groups: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for row in tqdm(ds, desc="Grouping by speaker"):
        name_field = row.get("name", "")
        speaker_id = parse_speaker_id(name_field)
        if speaker_id is None:
            skipped += 1
            continue
        groups[speaker_id].append(row)

    logger.info(f"Found {len(groups)} speakers | skipped {skipped} unparseable rows")
    return dict(groups)


def iter_speaker_audio(rows: list[dict]) -> Iterator[tuple[np.ndarray, int]]:
    for row in rows:
        audio = row.get("audio", {})
        arr = audio.get("array")
        sr = audio.get("sampling_rate")
        if arr is None or sr is None:
            continue
        yield np.asarray(arr, dtype=np.float32), int(sr)


def load_phonetic_dict(dataset_name: str) -> dict[str, str]:
    logger.info(f"Loading phonetic dictionary: {dataset_name}")
    ds = load_dataset(dataset_name, trust_remote_code=True)
    split = ds[list(ds.keys())[0]]

    mapping: dict[str, str] = {}
    for row in split:
        word = row.get("word") or row.get("text") or ""
        phonetic = row.get("phonetic") or row.get("ipa") or row.get("pronunciation") or ""
        if word and phonetic:
            mapping[word.strip()] = phonetic.strip()

    logger.info(f"Loaded {len(mapping)} phonetic entries")
    return mapping
