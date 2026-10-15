"""Export validated, cropped, isotropic CT-mask inputs for CERR radiomics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy import ndimage
from scipy.io import loadmat, savemat, whosmat
from tqdm import tqdm


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_MATCHED_ROOT = PROJECT_ROOT / "outputs_final" / "ct_mask_matched"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs_final" / "radiomics_inputs_mat"
DEFAULT_SESSION_MANIFEST = CODE_ROOT / "config" / "session_series_manifest.csv"


@dataclass
class ExportResult:
    patient_id: str
    status: str
    message: str
    input_path: str
    output_path: str
    source_shape_zyx: tuple[int, int, int] | None
    cropped_shape_zyx: tuple[int, int, int] | None
    mask_voxels: int | None
    source_volume_ml: float | None
    output_volume_ml: float | None
    volume_change_percent: float | None
    output_size_mb: float | None


def patient_sort_key(name: str) -> tuple[str, int, str]:
    """Sort patient IDs by their number."""
    prefix = "".join(char for char in name if not char.isdigit())
    digits = "".join(char for char in name if char.isdigit())
    return prefix, int(digits) if digits else -1, name


def load_session_manifest(path: Path) -> dict[str, str]:
    """Load the expected Series UID for each patient."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Session manifest is empty: {path}")

    manifest: dict[str, str] = {}
    for row in rows:
        patient_id = (row.get("patient_id") or "").strip()
        series_uid = (row.get("series_uid") or "").strip()
        status = (row.get("status") or "").strip().lower()
        if status != "ok" or not patient_id or not series_uid:
            raise ValueError(f"Invalid manifest row for {patient_id or 'unknown'}")
        if patient_id in manifest:
            raise ValueError(f"Duplicate patient in manifest: {patient_id}")
        manifest[patient_id] = series_uid
    return manifest


def load_metadata(path: Path) -> dict[str, Any]:
    """Load required spatial and provenance metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object")
    return metadata


def validate_metadata(
    metadata: dict[str, Any],
    patient_id: str,
    expected_series_uid: str,
    array_shape: tuple[int, int, int],
) -> None:
    """Check metadata needed by CERR."""
    if metadata.get("patient_id") != patient_id:
        raise ValueError("Patient ID differs between folder and metadata")

    dicom = metadata.get("dicom")
    if not isinstance(dicom, dict):
        raise ValueError("Missing DICOM metadata")
    if dicom.get("series_instance_uid") != expected_series_uid:
        raise ValueError("Series UID differs from Session.mat manifest")

    pixel_spacing = dicom.get("pixel_spacing_mm", dicom.get("pixel_spacing"))
    slice_spacing = dicom.get("slice_spacing_mm")
    if (
        not isinstance(pixel_spacing, (list, tuple))
        or len(pixel_spacing) < 2
        or any(float(value) <= 0 for value in pixel_spacing[:2])
    ):
        raise ValueError("Missing or invalid in-plane pixel spacing")
    if slice_spacing is None or float(slice_spacing) <= 0:
        raise ValueError("Missing or invalid physical slice spacing")

    metadata_ct_shape = tuple(int(value) for value in metadata["ct_shape_zyx"])
    metadata_mask_shape = tuple(
        int(value) for value in metadata["mask_shape_zyx"]
    )
    if metadata_ct_shape != array_shape or metadata_mask_shape != array_shape:
        raise ValueError("Array shape differs from metadata")
    if metadata.get("mask_alignment", {}).get("transpose_axes") != [2, 1, 0]:
        raise ValueError("Unexpected mask alignment")


def load_and_validate_arrays(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one CT-mask pair and validate its contents."""
    with np.load(npz_path, allow_pickle=False) as data:
        missing = {"ct_hu", "mask"} - set(data.files)
        if missing:
            raise ValueError(f"Missing NPZ arrays: {', '.join(sorted(missing))}")
        ct_hu = np.asarray(data["ct_hu"], dtype=np.float32)
        mask = np.asarray(data["mask"])

    if ct_hu.ndim != 3 or mask.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays: CT={ct_hu.shape}, mask={mask.shape}"
        )
    if ct_hu.shape != mask.shape:
        raise ValueError(f"Shape mismatch: CT={ct_hu.shape}, mask={mask.shape}")
    if not np.isfinite(ct_hu).all():
        raise ValueError("CT contains NaN or infinite values")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("Mask contains values other than 0 and 1")

    mask = mask.astype(np.uint8, copy=False)
    if not mask.any():
        raise ValueError("Mask is empty")
    return ct_hu, mask


def crop_to_mask(
    ct_hu: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Crop CT and mask to the prostate bounding box."""
    coordinates = np.nonzero(mask)
    starts = [int(axis.min()) for axis in coordinates]
    stops = [int(axis.max()) + 1 for axis in coordinates]
    crop = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    bounds = [[start, stop] for start, stop in zip(starts, stops)]
    return ct_hu[crop], mask[crop], bounds


def spacing_zyx(metadata: dict[str, Any]) -> tuple[float, float, float]:
    """Return array-axis spacing in z, y, x order."""
    dicom = metadata["dicom"]
    pixel = dicom.get("pixel_spacing_mm", dicom.get("pixel_spacing"))
    return float(dicom["slice_spacing_mm"]), float(pixel[0]), float(pixel[1])


def resample_to_isotropic(
    ct_hu: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, Any],
    target_mm: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Resample CT linearly and mask with nearest-neighbour interpolation."""
    source_spacing = spacing_zyx(metadata)
    source_volume = float(mask.sum() * np.prod(source_spacing) / 1000.0)
    zoom = tuple(spacing / target_mm for spacing in source_spacing)
    ct_new = ndimage.zoom(ct_hu, zoom=zoom, order=1, mode="nearest", prefilter=False)
    mask_new = ndimage.zoom(mask, zoom=zoom, order=0, mode="nearest", prefilter=False).astype(bool)
    ct_new, mask_new, _ = crop_to_mask(ct_new, mask_new)
    output_volume = float(mask_new.sum() * target_mm**3 / 1000.0)
    volume_change = 100.0 * (output_volume - source_volume) / source_volume

    dicom = metadata["dicom"]
    original_spacing = {
        "pixel_spacing_mm": dicom.get("pixel_spacing_mm", dicom.get("pixel_spacing")),
        "slice_spacing_mm": dicom.get("slice_spacing_mm"),
        "slice_thickness_mm": dicom.get("slice_thickness_mm"),
    }
    dicom["pixel_spacing_mm"] = [target_mm, target_mm]
    dicom["pixel_spacing"] = [target_mm, target_mm]
    dicom["slice_spacing_mm"] = target_mm
    dicom["slice_thickness_mm"] = target_mm
    metadata["radiomics_input_resampling"] = {
        "applied": True,
        "target_spacing_zyx_mm": [target_mm] * 3,
        "source_spacing_zyx_mm": list(source_spacing),
        "ct_interpolation": "linear",
        "mask_interpolation": "nearest",
        "original_dicom_spacing": original_spacing,
        "source_shape_zyx": list(ct_hu.shape),
        "output_shape_zyx": list(ct_new.shape),
        "source_mask_volume_ml": source_volume,
        "output_mask_volume_ml": output_volume,
        "volume_change_percent": volume_change,
    }
    return ct_new, mask_new, source_volume, output_volume, volume_change

def read_mat_metadata(path: Path) -> dict[str, Any]:
    """Read only metadata_json from a MAT file."""
    loaded = loadmat(
        path,
        variable_names=["metadata_json"],
        squeeze_me=True,
        chars_as_strings=True,
    )
    value = loaded.get("metadata_json")
    if value is None:
        raise ValueError("MAT file is missing metadata_json")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.item()
        else:
            value = "".join(str(item) for item in value.flat)
    return json.loads(str(value))


def validate_mat_file(
    path: Path,
    patient_id: str,
    expected_series_uid: str,
    expected_shape: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    """Check MAT variables, shapes, and provenance without loading CT pixels."""
    variables = {name: tuple(shape) for name, shape, _ in whosmat(path)}
    required = {"patient_id", "ct_hu", "mask", "metadata_json"}
    missing = required - set(variables)
    if missing:
        raise ValueError(f"Incomplete MAT file: {', '.join(sorted(missing))}")

    ct_shape = variables["ct_hu"]
    mask_shape = variables["mask"]
    if len(ct_shape) != 3 or ct_shape != mask_shape:
        raise ValueError(
            f"Invalid MAT array shapes: CT={ct_shape}, mask={mask_shape}"
        )
    if expected_shape is not None and ct_shape != expected_shape:
        raise ValueError(
            f"MAT shape differs from new export: {ct_shape} != {expected_shape}"
        )

    metadata = read_mat_metadata(path)
    export = metadata.get("radiomics_input", {})
    if metadata.get("patient_id") != patient_id:
        raise ValueError("MAT patient ID does not match its filename")
    if export.get("source_series_uid") != expected_series_uid:
        raise ValueError("MAT Series UID differs from Session.mat manifest")
    return ct_shape


def save_mat_safely(
    output_path: Path,
    contents: dict[str, Any],
    patient_id: str,
    expected_series_uid: str,
    expected_shape: tuple[int, int, int],
    compressed: bool,
) -> None:
    """Write and verify a temporary MAT file before replacing the output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}.tmp.mat"
    )
    try:
        savemat(
            temporary_path,
            contents,
            do_compression=compressed,
            appendmat=False,
        )
        validate_mat_file(
            temporary_path,
            patient_id,
            expected_series_uid,
            expected_shape,
        )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def convert_one(
    patient_id: str,
    expected_series_uid: str,
    matched_root: Path,
    output_root: Path,
    overwrite: bool,
    compressed: bool,
    target_mm: float,
) -> ExportResult:
    """Validate and export one patient."""
    patient_dir = matched_root / patient_id
    npz_path = patient_dir / "data.npz"
    metadata_path = patient_dir / "metadata.json"
    output_path = output_root / f"{patient_id}.mat"

    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")

    if output_path.exists() and not overwrite:
        shape = validate_mat_file(
            output_path,
            patient_id,
            expected_series_uid,
        )
        return ExportResult(
            patient_id=patient_id,
            status="verified_existing",
            message="existing MAT file is complete",
            input_path=str(npz_path),
            output_path=str(output_path),
            source_shape_zyx=None,
            cropped_shape_zyx=shape,
            mask_voxels=None,
            source_volume_ml=None,
            output_volume_ml=None,
            volume_change_percent=None,
            output_size_mb=output_path.stat().st_size / (1024**2),
        )

    ct_hu, mask = load_and_validate_arrays(npz_path)
    source_shape = tuple(int(value) for value in ct_hu.shape)
    metadata = load_metadata(metadata_path)
    validate_metadata(
        metadata,
        patient_id,
        expected_series_uid,
        source_shape,
    )

    ct_crop, mask_crop, crop_bounds = crop_to_mask(ct_hu, mask)
    pre_resample_shape = tuple(int(value) for value in ct_crop.shape)
    metadata["radiomics_input"] = {
        "source_series_uid": expected_series_uid,
        "source_npz": str(npz_path),
        "source_shape_zyx": source_shape,
        "cropped_shape_zyx": pre_resample_shape,
        "crop_bounds_zyx": crop_bounds,
        "cropped_to_mask_bounding_box": True,
    }
    ct_crop, mask_crop, source_volume, output_volume, volume_change = (
        resample_to_isotropic(ct_crop, mask_crop, metadata, target_mm)
    )
    cropped_shape = tuple(int(value) for value in ct_crop.shape)
    mask_voxels = int(mask_crop.sum())

    save_mat_safely(
        output_path=output_path,
        contents={
            "patient_id": patient_id,
            "ct_hu": np.ascontiguousarray(ct_crop, dtype=np.float32),
            "mask": np.ascontiguousarray(mask_crop, dtype=np.uint8),
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
        },
        patient_id=patient_id,
        expected_series_uid=expected_series_uid,
        expected_shape=cropped_shape,
        compressed=compressed,
    )
    return ExportResult(
        patient_id=patient_id,
        status="converted",
        message=f"validated, cropped, and resampled to {target_mm:g} mm",
        input_path=str(npz_path),
        output_path=str(output_path),
        source_shape_zyx=source_shape,
        cropped_shape_zyx=cropped_shape,
        mask_voxels=mask_voxels,
        source_volume_ml=source_volume,
        output_volume_ml=output_volume,
        volume_change_percent=volume_change,
        output_size_mb=output_path.stat().st_size / (1024**2),
    )


def write_reports(
    results: list[ExportResult],
    output_root: Path,
    report_name: str,
) -> None:
    """Write CSV and JSON export reports."""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{report_name}.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    with (output_root / f"{report_name}.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(ExportResult.__dataclass_fields__),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def parse_args() -> argparse.Namespace:
    """Read command line options."""
    parser = argparse.ArgumentParser(
        description="Export validated, cropped, isotropic CT-mask MAT files."
    )
    parser.add_argument("--matched-root", type=Path, default=DEFAULT_MATCHED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--session-manifest",
        type=Path,
        default=DEFAULT_SESSION_MANIFEST,
    )
    parser.add_argument("--patients", nargs="*")
    parser.add_argument("--report-name", default="mat_export_resampling_report")
    parser.add_argument("--target-spacing-mm", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable MAT compression.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the MATLAB input export."""
    args = parse_args()
    if args.target_spacing_mm <= 0:
        raise ValueError("Target spacing must be positive")
    manifest = load_session_manifest(args.session_manifest)
    patients = (
        sorted(args.patients, key=patient_sort_key)
        if args.patients
        else sorted(manifest, key=patient_sort_key)
    )
    missing = [patient_id for patient_id in patients if patient_id not in manifest]
    if missing:
        raise ValueError(f"Patients missing from manifest: {', '.join(missing)}")

    print(f"Matched root: {args.matched_root.resolve()}")
    print(f"Output root: {args.output_root.resolve()}")
    print(f"Patients: {len(patients)}")

    results: list[ExportResult] = []
    for patient_id in tqdm(patients, desc="MAT export"):
        try:
            result = convert_one(
                patient_id=patient_id,
                expected_series_uid=manifest[patient_id],
                matched_root=args.matched_root,
                output_root=args.output_root,
                overwrite=args.overwrite,
                compressed=not args.no_compression,
                target_mm=args.target_spacing_mm,
            )
        except Exception as error:
            result = ExportResult(
                patient_id=patient_id,
                status="error",
                message=f"{type(error).__name__}: {error}",
                input_path=str(args.matched_root / patient_id / "data.npz"),
                output_path=str(args.output_root / f"{patient_id}.mat"),
                source_shape_zyx=None,
                cropped_shape_zyx=None,
                mask_voxels=None,
                source_volume_ml=None,
                output_volume_ml=None,
                volume_change_percent=None,
                output_size_mb=None,
            )
        results.append(result)

    write_reports(results, args.output_root, args.report_name)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print(f"Summary: {counts}")
    print(f"Report: {args.output_root / f'{args.report_name}.csv'}")
    if counts.get("error", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
