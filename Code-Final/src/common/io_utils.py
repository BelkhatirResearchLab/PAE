"""Small input/output helpers shared by the final scripts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from uuid import uuid4

import pandas as pd


def normalise_patient_id(value: object) -> str:
    """Convert common patient-ID forms to PAE<number>."""
    text = str(value).strip().upper()
    match = re.search(r"PAE\s*[-_ ]*\s*(\d+)", text)
    if match:
        return f"PAE{int(match.group(1))}"
    if re.fullmatch(r"\d+(\.0+)?", text):
        return f"PAE{int(float(text))}"
    return text


def require_unique_ids(table: pd.DataFrame, name: str) -> None:
    """Reject blank or duplicate patient IDs."""
    if "patient_id" not in table:
        raise ValueError(f"{name} has no patient_id column")
    if table["patient_id"].eq("").any():
        raise ValueError(f"{name} contains blank patient IDs")
    duplicates = table.loc[table["patient_id"].duplicated(), "patient_id"].tolist()
    if duplicates:
        raise ValueError(f"{name} contains duplicate IDs: {duplicates}")


def atomic_csv(table: pd.DataFrame, path: Path) -> None:
    """Write a CSV only after the complete temporary file is ready."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        table.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(value: dict, path: Path) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(value: str, path: Path) -> None:
    """Write UTF-8 text atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

