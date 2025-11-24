import logging
from pathlib import Path
from string import Template
from typing import Iterable

import pandas as pd


def read_excel_file(filepath: str | Path) -> pd.DataFrame:
    """
    Reads an Excel file and returns a DataFrame.
    Assumes that the headers are in row 1 (index 0) and data starts from row 2 (index 1).
    """
    return pd.read_excel(filepath, header=0)


def load_txt(path: str | Path) -> str:
    """
    Read a UTF-8 text file and return its content.
    """
    path = Path(path).expanduser()
    return path.read_text(encoding="utf-8")


def load_template(path: str | Path, *, required: Iterable[str] | None = None) -> Template:
    """
    Read a file and wrap it in :class:`string.Template`.

    Parameters
    ----------
    required
        Placeholder names (including the leading ``$``) that **must** appear.
    """
    text = load_txt(path)
    if required:
        missing = [ph for ph in required if ph not in text]
        if missing:
            raise ValueError(f"{path} missing placeholders: {', '.join(missing)}")
    return Template(text)


def load_guidelines_text(instructions_root: Path, language: str, filename: str | None = None) -> str:
    """
    Load the language-specific guidelines text.
    If filename is not provided, the first .txt file in the language folder is used.
    """
    lang_dir = instructions_root / language
    if not lang_dir.exists():
        raise FileNotFoundError(f"Guidelines folder not found for language '{language}': {lang_dir}")

    if filename:
        file_path = lang_dir / filename
    else:
        txt_files = list(lang_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt guideline files found in {lang_dir}")
        file_path = txt_files[0]

    if not file_path.exists():
        raise FileNotFoundError(f"Guidelines file not found: {file_path}")

    logging.info("Loading guidelines for %s from %s", language, file_path)
    return load_txt(file_path).strip()
