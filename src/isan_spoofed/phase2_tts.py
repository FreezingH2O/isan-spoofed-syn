from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger
from openai import OpenAI
from tqdm import tqdm


def _make_client(tts_cfg: dict) -> OpenAI:
    return OpenAI(
        base_url=tts_cfg["base_url"],
        api_key=tts_cfg["api_key"],
    )


def synthesize(text: str, out_path: Path, tts_cfg: dict, voice: Optional[str] = None) -> Path:
    client = _make_client(tts_cfg)
    voice = voice or tts_cfg["default_voice"]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model=tts_cfg["model"],
        voice=voice,
        input=text,
        response_format=tts_cfg["output_format"],
    ) as response:
        response.stream_to_file(str(out_path))

    return out_path


def run(cfg: dict, speaker_id: Optional[str] = None) -> list[Path]:
    tts_cfg: dict = cfg["tts"]
    norm_dir = Path(cfg["paths"]["data"]["normalized_text"])
    out_dir = Path(cfg["paths"]["data"]["tts_output"])
    out_dir.mkdir(parents=True, exist_ok=True)

    text_files = sorted(norm_dir.glob("*.normalized.txt"))
    if not text_files:
        logger.warning(f"No normalized text files found in {norm_dir}. Run Phase 1 first.")
        return []

    written: list[Path] = []
    for txt_path in tqdm(text_files, desc="TTS synthesis"):
        stem = txt_path.name.replace(".normalized.txt", "")
        out_path = out_dir / f"{stem}.wav"

        if out_path.exists():
            logger.debug(f"  Skipping {stem} (already exists)")
            written.append(out_path)
            continue

        raw = txt_path.read_text(encoding="utf-8").strip()
        if not raw:
            logger.warning(f"  Skipping empty file: {txt_path.name}")
            continue

        # OPTIONAL VOICE OVERRIDE: #voice: s3 ON FIRST LINE
        voice: Optional[str] = None
        if raw.startswith("#voice:"):
            first_line, _, raw = raw.partition("\n")
            voice = first_line.split(":", 1)[1].strip()
            raw = raw.strip()

        try:
            synthesize(raw, out_path, tts_cfg, voice=voice)
            logger.info(f"  {stem} [{voice or tts_cfg['default_voice']}] → {out_path.name}")
            written.append(out_path)
        except Exception as exc:
            logger.error(f"  Failed to synthesize {stem}: {exc}")

    _write_manifest(out_dir, written)
    logger.success(f"Phase 2 complete: {len(written)} wav files → {out_dir}")
    return written


def _write_manifest(out_dir: Path, paths: list[Path]) -> None:
    manifest = [{"file": p.name} for p in paths]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
