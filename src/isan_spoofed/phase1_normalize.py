from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from isan_spoofed.utils.dataset import load_phonetic_dict


# THAI WORD TABLES FOR NUMBER CONVERSION
_ONES = ["ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]
_TENS = ["", "", "ยี่สิบ", "สามสิบ", "สี่สิบ", "ห้าสิบ", "หกสิบ", "เจ็ดสิบ", "แปดสิบ", "เก้าสิบ"]


# CONVERT INTEGER TO THAI WORDS (RECURSIVE)
def _int_to_thai(n: int) -> str:
    if n < 0:
        return "ลบ" + _int_to_thai(-n)
    if n < 10:
        return _ONES[n]
    if n < 20:
        suffix = "เอ็ด" if n % 10 == 1 else (_ONES[n % 10] if n % 10 != 0 else "")
        return "สิบ" + suffix
    if n < 100:
        unit = _ONES[n % 10] if n % 10 != 0 else ""
        return _TENS[n // 10] + unit
    if n < 1_000:
        rest = _int_to_thai(n % 100) if n % 100 != 0 else ""
        return _ONES[n // 100] + "ร้อย" + rest
    if n < 10_000:
        rest = _int_to_thai(n % 1_000) if n % 1_000 != 0 else ""
        return _ONES[n // 1_000] + "พัน" + rest
    if n < 100_000:
        rest = _int_to_thai(n % 10_000) if n % 10_000 != 0 else ""
        return _int_to_thai(n // 10_000) + "หมื่น" + rest
    if n < 1_000_000:
        rest = _int_to_thai(n % 100_000) if n % 100_000 != 0 else ""
        return _int_to_thai(n // 100_000) + "แสน" + rest
    rest = _int_to_thai(n % 1_000_000) if n % 1_000_000 != 0 else ""
    return _int_to_thai(n // 1_000_000) + "ล้าน" + rest


def _replace_numbers(text: str) -> str:
    def _replace(m: re.Match) -> str:
        return _int_to_thai(int(m.group()))
    return re.sub(r"\d+", _replace, text)


# INSERT \n PHRASE BREAKS FOR F5-TTS PROSODY
_PHRASE_BREAK_RE = re.compile(r"([ๆ฿฀-๿]+[.!?।])\s+")


def _insert_phrase_breaks(text: str) -> str:
    return _PHRASE_BREAK_RE.sub(r"\1\n", text)


def _apply_phonetic_dict(text: str, phonetic_dict: dict[str, str]) -> tuple[str, int]:
    tokens = text.split()
    resolved: list[str] = []
    substituted = 0
    for token in tokens:
        clean = token.strip(".,!?;:\"'")
        if clean in phonetic_dict:
            resolved.append(phonetic_dict[clean])
            substituted += 1
        else:
            resolved.append(token)
    return " ".join(resolved), substituted


def normalize_text(text: str, phonetic_dict: dict[str, str]) -> str:
    text = _replace_numbers(text)
    text, substituted = _apply_phonetic_dict(text, phonetic_dict)
    text = _insert_phrase_breaks(text)
    logger.debug(f"Normalized: {substituted} phonetic substitutions applied")
    return text.strip()


def _read_multiline_stdin() -> str:
    # READ UNTIL BLANK LINE OR EOF
    print("  (Enter text, then press Enter twice or Ctrl+D to finish)")
    lines = []
    try:
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines).strip()


def _collect_sources(
    cfg: dict,
    text: Optional[str] = None,
    input_path: Optional[str] = None,
) -> dict[str, str]:
    out_dir = Path(cfg["paths"]["data"]["normalized_text"])

    # EXPLICIT TEXT STRING FROM CLI OR MENU
    if text:
        return {"utterance_0001": text.strip()}

    # EXPLICIT FILE OR DIRECTORY PATH
    if input_path:
        p = Path(input_path)
        if p.is_file():
            return {p.stem: p.read_text(encoding="utf-8")}
        if p.is_dir():
            files = sorted(p.glob("*.txt"))
            if not files:
                logger.warning(f"No .txt files found in {p}")
                return {}
            return {f.stem: f.read_text(encoding="utf-8") for f in files}
        logger.error(f"Path not found: {p}")
        return {}

    # DEFAULT: data/normalized_text/input/
    input_dir = out_dir / "input"
    if input_dir.exists() and any(input_dir.glob("*.txt")):
        return {p.stem: p.read_text(encoding="utf-8") for p in sorted(input_dir.glob("*.txt"))}

    logger.warning(f"No input found in {input_dir} — use --text or --input-path, or drop .txt files there.")
    return {}


def run(
    cfg: dict,
    speaker_id: Optional[str] = None,
    text: Optional[str] = None,
    input_path: Optional[str] = None,
    input_texts: Optional[list[str]] = None,
) -> list[Path]:
    dict_name: str = cfg["datasets"]["phonetic_dict"]
    out_dir = Path(cfg["paths"]["data"]["normalized_text"])
    out_dir.mkdir(parents=True, exist_ok=True)

    phonetic_dict = load_phonetic_dict(dict_name)

    # PROGRAMMATIC OVERRIDE (USED BY TESTS / OTHER PHASES)
    if input_texts is not None:
        sources = {f"utterance_{i:04d}": t for i, t in enumerate(input_texts)}
    else:
        sources = _collect_sources(cfg, text=text, input_path=input_path)

    if not sources:
        return []

    written: list[Path] = []
    for name, raw in sources.items():
        normalized = normalize_text(raw, phonetic_dict)
        out_path = out_dir / f"{name}.normalized.txt"
        out_path.write_text(normalized, encoding="utf-8")
        logger.info(f"  {name} → {out_path.name}")
        written.append(out_path)

    _write_manifest(out_dir, written)
    logger.success(f"Phase 1 complete: {len(written)} texts normalized → {out_dir}")
    return written


def _write_manifest(out_dir: Path, paths: list[Path]) -> None:
    manifest = [{"file": p.name} for p in paths]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
