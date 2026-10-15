"""Load the CT series recorded in each Medical Image Labeler session."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from tqdm import tqdm


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs_final" / "ct_mask_matched"
DEFAULT_SESSION_MANIFEST = CODE_ROOT / "config" / "session_series_manifest.csv"


@dataclass(frozen=True)
class SessionSeries:
    patient_id: str
    series_uid: str
    series_folder: str
    source_file_count: int


@dataclass
class SliceRecord:
    path: Path
    header: Any | None = None


@dataclass
class PatientResult:
    patient_id: str
    status: str
    message: str
    mask_path: str | None
    mask_shape_xyz: tuple[int, int, int] | None
    selected_series_uid: str | None
    selected_series_dir: str | None
    ct_shape_zyx: tuple[int, int, int] | None
    mask_shape_zyx: tuple[int, int, int] | None
    mask_voxels: int | None
    mask_volume_ml: float | None


def win_path(path: Path) -> str:
    """Return a Windows long path."""
    absolute = str(path.resolve())
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def patient_sort_key(name: str) -> tuple[str, int, str]:
    """Sort patient IDs by their number."""
    prefix = "".join(char for char in name if not char.isdigit())
    digits = "".join(char for char in name if char.isdigit())
    return prefix, int(digits) if digits else -1, name


def load_session_manifest(path: Path) -> dict[str, SessionSeries]:
    """Load the CT series extracted from Session.mat files."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "patient_id",
        "series_uid",
        "series_folder",
        "source_file_count",
        "status",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid session manifest: {path}")

    manifest: dict[str, SessionSeries] = {}
    for row in rows:
        patient_id = (row["patient_id"] or "").strip()
        status = (row["status"] or "").strip().lower()
        if status != "ok":
            raise ValueError(
                f"Session manifest error for {patient_id}: {row['status']}"
            )
        if patient_id in manifest:
            raise ValueError(f"Duplicate patient in manifest: {patient_id}")

        entry = SessionSeries(
            patient_id=patient_id,
            series_uid=(row["series_uid"] or "").strip(),
            series_folder=(row["series_folder"] or "").strip(),
            source_file_count=int(row["source_file_count"]),
        )
        if not entry.patient_id or not entry.series_uid or not entry.series_folder:
            raise ValueError(f"Incomplete manifest row for {patient_id}")
        manifest[patient_id] = entry
    return manifest


def read_dicom_header(path: Path) -> Any:
    """Read one DICOM header without pixel data."""
    try:
        return pydicom.dcmread(
            win_path(path),
            stop_before_pixels=True,
            force=False,
        )
    except (InvalidDicomError, OSError, PermissionError) as error:
        raise ValueError(f"Cannot read DICOM header: {path.name}") from error


def resolve_series(
    patient_dir: Path,
    session_series: SessionSeries,
) -> tuple[Path, list[SliceRecord]]:
    """Open only the series folder recorded in Session.mat."""
    series_dir = patient_dir / session_series.series_folder
    if not series_dir.is_dir():
        raise FileNotFoundError(f"Series folder not found: {series_dir}")

    files = sorted(Path(win_path(series_dir)).rglob("*.dcm"))
    if len(files) != session_series.source_file_count:
        raise ValueError(
            "DICOM file count differs from Session.mat: "
            f"{len(files)} != {session_series.source_file_count}"
        )
    return series_dir, [SliceRecord(path) for path in files]


def validate_manifest_match(
    records: list[SliceRecord],
    session_series: SessionSeries,
    mask_shape_xyz: tuple[int, int, int],
) -> None:
    """Check the selected series against Session.mat and the mask."""
    if not records:
        raise ValueError("Selected DICOM series is empty")

    header = read_dicom_header(records[0].path)
    records[0].header = header
    series_uid = str(getattr(header, "SeriesInstanceUID", "") or "")
    modality = str(getattr(header, "Modality", "") or "").upper()
    rows = int(getattr(header, "Rows", 0) or 0)
    cols = int(getattr(header, "Columns", 0) or 0)

    if series_uid != session_series.series_uid:
        raise ValueError(
            f"Series UID differs from Session.mat: {series_uid}"
        )
    if modality != "CT":
        raise ValueError(f"Selected series is not CT: {modality or 'missing'}")
    if len(records) != mask_shape_xyz[2]:
        raise ValueError(
            f"Depth mismatch: CT={len(records)}, mask={mask_shape_xyz[2]}"
        )
    if rows != mask_shape_xyz[1] or cols != mask_shape_xyz[0]:
        raise ValueError(
            f"In-plane mismatch: CT={rows}x{cols}, "
            f"mask={mask_shape_xyz[1]}x{mask_shape_xyz[0]}"
        )


def get_slice_position(header: Any) -> float | None:
    """Get the physical position along the slice axis."""
    orientation = getattr(header, "ImageOrientationPatient", None)
    position = getattr(header, "ImagePositionPatient", None)
    if orientation is None or position is None:
        return None
    if len(orientation) < 6 or len(position) < 3:
        return None

    row_direction = np.asarray(orientation[:3], dtype=float)
    column_direction = np.asarray(orientation[3:6], dtype=float)
    slice_normal = np.cross(row_direction, column_direction)
    return float(np.dot(np.asarray(position[:3], dtype=float), slice_normal))


def sort_slice_records(records: list[SliceRecord]) -> list[SliceRecord]:
    """Sort slices by physical position."""
    positions = [get_slice_position(record.header) for record in records]
    if all(position is not None for position in positions):
        pairs = zip(positions, records)
        return [
            record
            for _, record in sorted(pairs, key=lambda item: float(item[0]))
        ]

    # Use Instance Number only when position is missing.
    return sorted(
        records,
        key=lambda record: int(
            getattr(record.header, "InstanceNumber", 0) or 0
        ),
    )


def compact_header(dataset: Any) -> SimpleNamespace:
    """Keep only fields needed after pixel loading."""
    fields = (
        "SeriesInstanceUID",
        "StudyInstanceUID",
        "FrameOfReferenceUID",
        "SeriesDescription",
        "Rows",
        "Columns",
        "PixelSpacing",
        "SliceThickness",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "InstanceNumber",
    )
    return SimpleNamespace(
        **{
            field: getattr(dataset, field)
            for field in fields
            if hasattr(dataset, field)
        }
    )


def validate_series(
    records: list[SliceRecord],
) -> tuple[list[SliceRecord], float]:
    """Check slice geometry and return the slice spacing."""
    if any(record.header is None for record in records):
        raise ValueError("Missing DICOM headers")

    records = sort_slice_records(records)
    first = records[0].header
    expected_uid = str(first.SeriesInstanceUID)
    expected_rows = int(first.Rows)
    expected_cols = int(first.Columns)
    expected_spacing = np.asarray(first.PixelSpacing, dtype=float)
    expected_orientation = np.asarray(first.ImageOrientationPatient, dtype=float)

    for record in records:
        header = record.header
        if str(header.SeriesInstanceUID) != expected_uid:
            raise ValueError("Mixed Series Instance UIDs")
        if int(header.Rows) != expected_rows or int(header.Columns) != expected_cols:
            raise ValueError("Inconsistent slice matrix")
        if not np.allclose(
            np.asarray(header.PixelSpacing, dtype=float),
            expected_spacing,
            atol=1e-4,
        ):
            raise ValueError("Inconsistent pixel spacing")
        if not np.allclose(
            np.asarray(header.ImageOrientationPatient, dtype=float),
            expected_orientation,
            atol=1e-4,
        ):
            raise ValueError("Inconsistent slice orientation")

    positions = [get_slice_position(record.header) for record in records]
    if any(position is None for position in positions):
        raise ValueError("Missing physical slice positions")

    gaps = np.abs(np.diff(np.asarray(positions, dtype=float)))
    if gaps.size == 0:
        raise ValueError("Series has fewer than two slices")
    if np.any(gaps < 1e-3):
        raise ValueError("Duplicate slice positions")

    slice_spacing = float(np.median(gaps))
    tolerance = max(0.1, 0.05 * slice_spacing)
    if np.any(np.abs(gaps - slice_spacing) > tolerance):
        raise ValueError("Irregular slice spacing")
    return records, slice_spacing


def load_ct_hu(records: list[SliceRecord]) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and validate one CT series in a single pass."""
    arrays: dict[Path, np.ndarray] = {}
    for record in records:
        dataset = pydicom.dcmread(win_path(record.path), force=False)
        if not hasattr(dataset, "PixelData"):
            raise ValueError(f"Missing PixelData: {record.path.name}")

        pixels = dataset.pixel_array.astype(np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
        pixels *= slope
        pixels += intercept
        arrays[record.path] = pixels
        record.header = compact_header(dataset)

    records, slice_spacing = validate_series(records)
    first = records[0].header
    pixel_spacing = tuple(float(value) for value in first.PixelSpacing)
    ct_hu = np.stack([arrays[record.path] for record in records], axis=0)
    metadata = {
        "series_instance_uid": str(first.SeriesInstanceUID),
        "study_instance_uid": str(getattr(first, "StudyInstanceUID", "") or ""),
        "frame_of_reference_uid": str(
            getattr(first, "FrameOfReferenceUID", "") or ""
        ),
        "series_description": str(
            getattr(first, "SeriesDescription", "") or ""
        ),
        "rows": int(first.Rows),
        "cols": int(first.Columns),
        "n_slices": len(records),
        "pixel_spacing": pixel_spacing,
        "slice_thickness": slice_spacing,
        "pixel_spacing_mm": pixel_spacing,
        "slice_spacing_mm": slice_spacing,
        "slice_thickness_mm": float(
            getattr(first, "SliceThickness", 0.0) or 0.0
        ),
        "image_orientation_patient": [
            float(value) for value in first.ImageOrientationPatient
        ],
        "image_position_first": [
            float(value) for value in records[0].header.ImagePositionPatient
        ],
        "image_position_last": [
            float(value) for value in records[-1].header.ImagePositionPatient
        ],
    }
    return ct_hu, metadata


def align_mask(mask_xyz: np.ndarray) -> np.ndarray:
    """Convert the mask from XYZ to CT ZYX order."""
    if mask_xyz.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got {mask_xyz.shape}")
    return np.transpose(mask_xyz > 0.5, (2, 1, 0)).astype(np.uint8)


def calculate_mask_volume_ml(
    mask: np.ndarray,
    pixel_spacing_mm: tuple[float, float],
    slice_spacing_mm: float,
) -> float:
    """Calculate mask volume in millilitres."""
    voxel_volume_mm3 = (
        pixel_spacing_mm[0] * pixel_spacing_mm[1] * slice_spacing_mm
    )
    return float(mask.sum() * voxel_volume_mm3 / 1000.0)


def save_qc_overlay(
    ct_hu: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
) -> None:
    """Save axial, coronal, and sagittal mask overlays."""
    coordinates = np.argwhere(mask > 0)
    if coordinates.size == 0:
        raise ValueError("Mask is empty")

    z, y, x = np.round(coordinates.mean(axis=0)).astype(int)
    views = [
        ("Axial", ct_hu[z], mask[z]),
        ("Coronal", ct_hu[:, y, :], mask[:, y, :]),
        ("Sagittal", ct_hu[:, :, x], mask[:, :, x]),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (title, image, label) in zip(axes, views):
        axis.imshow(image, cmap="gray", vmin=-200, vmax=400)
        axis.contour(label, levels=[0.5], colors="red", linewidths=0.8)
        axis.set_title(title)
        axis.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def validate_existing_output(
    patient_output: Path,
    session_series: SessionSeries,
    mask_shape_xyz: tuple[int, int, int],
    require_qc: bool,
) -> dict[str, Any]:
    """Verify an existing export against the Session.mat manifest."""
    required_files = ["data.npz", "metadata.json"]
    if require_qc:
        required_files.append("qc_overlay.png")
    missing = [
        name
        for name in required_files
        if not (patient_output / name).is_file()
    ]
    if missing:
        raise ValueError(f"Existing output is incomplete: {', '.join(missing)}")

    metadata = json.loads(
        (patient_output / "metadata.json").read_text(encoding="utf-8")
    )
    existing_uid = str(metadata["dicom"]["series_instance_uid"])
    existing_slices = int(metadata["dicom"]["n_slices"])
    existing_mask_shape = tuple(metadata["mask_original_shape_xyz"])
    transpose_axes = metadata.get("mask_alignment", {}).get("transpose_axes")

    if existing_uid != session_series.series_uid:
        raise ValueError(
            "Existing output has a different Series UID; "
            "rerun this patient with --overwrite"
        )
    if existing_slices != session_series.source_file_count:
        raise ValueError(
            "Existing output has a different slice count; "
            "rerun this patient with --overwrite"
        )
    if existing_mask_shape != mask_shape_xyz:
        raise ValueError(
            "Existing output has a different mask shape; "
            "rerun this patient with --overwrite"
        )
    if transpose_axes != [2, 1, 0]:
        raise ValueError(
            "Existing output used a different mask alignment; "
            "rerun this patient with --overwrite"
        )
    return metadata


def process_patient(
    patient_id: str,
    session_series: SessionSeries,
    manifest_path: Path,
    data_root: Path,
    output_root: Path,
    report_only: bool,
    overwrite: bool,
    make_qc: bool,
    compressed: bool,
) -> PatientResult:
    """Process one patient."""
    patient_dir = data_root / "DICOM" / patient_id
    mask_path = (
        data_root
        / "SAVED SESSIONS"
        / patient_id
        / "LabelData"
        / "files.nii"
    )
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Missing DICOM folder: {patient_dir}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    # Read the mask header.
    mask_image = nib.load(win_path(mask_path))
    mask_shape_xyz = tuple(int(value) for value in mask_image.shape)
    if len(mask_shape_xyz) != 3:
        raise ValueError(f"Expected a 3D mask, got {mask_shape_xyz}")

    patient_output = output_root / patient_id
    data_output = patient_output / "data.npz"
    if data_output.exists() and not overwrite and not report_only:
        metadata = validate_existing_output(
            patient_output,
            session_series,
            mask_shape_xyz,
            make_qc,
        )
        return PatientResult(
            patient_id=patient_id,
            status="skip",
            message="existing output verified against Session.mat",
            mask_path=str(mask_path),
            mask_shape_xyz=mask_shape_xyz,
            selected_series_uid=session_series.series_uid,
            selected_series_dir=session_series.series_folder,
            ct_shape_zyx=tuple(metadata["ct_shape_zyx"]),
            mask_shape_zyx=tuple(metadata["mask_shape_zyx"]),
            mask_voxels=int(metadata["mask_voxels"]),
            mask_volume_ml=float(metadata["mask_volume_ml"]),
        )

    # Open only the CT series recorded in Session.mat.
    series_dir, records = resolve_series(patient_dir, session_series)
    validate_manifest_match(records, session_series, mask_shape_xyz)
    if report_only:
        return PatientResult(
            patient_id=patient_id,
            status="matched",
            message="Session.mat series verified",
            mask_path=str(mask_path),
            mask_shape_xyz=mask_shape_xyz,
            selected_series_uid=session_series.series_uid,
            selected_series_dir=str(series_dir),
            ct_shape_zyx=None,
            mask_shape_zyx=None,
            mask_voxels=None,
            mask_volume_ml=None,
        )

    # Load CT and convert the mask to ZYX order.
    ct_hu, dicom_metadata = load_ct_hu(records)
    mask_xyz = mask_image.get_fdata(dtype=np.float32)
    mask = align_mask(mask_xyz)
    if ct_hu.shape != mask.shape:
        raise ValueError(f"Shape mismatch: CT={ct_hu.shape}, mask={mask.shape}")
    if not mask.any():
        raise ValueError("Mask is empty")

    pixel_spacing = tuple(dicom_metadata["pixel_spacing_mm"])
    slice_spacing = float(dicom_metadata["slice_spacing_mm"])
    mask_volume_ml = calculate_mask_volume_ml(
        mask,
        pixel_spacing,
        slice_spacing,
    )

    # Save arrays and QC.
    patient_output.mkdir(parents=True, exist_ok=True)
    save_function = np.savez_compressed if compressed else np.savez
    save_function(data_output, ct_hu=ct_hu, mask=mask)
    if make_qc:
        save_qc_overlay(ct_hu, mask, patient_output / "qc_overlay.png")

    # Save metadata for review and radiomics.
    metadata = {
        "patient_id": patient_id,
        "mask_path": str(mask_path),
        "mask_original_shape_xyz": mask_shape_xyz,
        "mask_shape_zyx": tuple(int(value) for value in mask.shape),
        "mask_alignment": {
            "method": "fixed_transpose",
            "transpose_axes": [2, 1, 0],
            "resampled": False,
        },
        "mask_voxels": int(mask.sum()),
        "mask_volume_ml": mask_volume_ml,
        "ct_shape_zyx": tuple(int(value) for value in ct_hu.shape),
        "ct_hu_min": float(ct_hu.min()),
        "ct_hu_max": float(ct_hu.max()),
        "dicom": dicom_metadata,
        "selected_series": asdict(session_series),
        "series_selection": {
            "method": "session_data_source",
            "manifest_path": str(manifest_path),
        },
    }
    (patient_output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return PatientResult(
        patient_id=patient_id,
        status="success",
        message="exported from Session.mat series",
        mask_path=str(mask_path),
        mask_shape_xyz=mask_shape_xyz,
        selected_series_uid=session_series.series_uid,
        selected_series_dir=str(series_dir),
        ct_shape_zyx=tuple(int(value) for value in ct_hu.shape),
        mask_shape_zyx=tuple(int(value) for value in mask.shape),
        mask_voxels=int(mask.sum()),
        mask_volume_ml=mask_volume_ml,
    )


def write_reports(
    results: list[PatientResult],
    output_root: Path,
    report_name: str,
) -> None:
    """Write JSON and CSV processing reports."""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{report_name}.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )

    fields = list(PatientResult.__dataclass_fields__)
    with (output_root / f"{report_name}.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def parse_args() -> argparse.Namespace:
    """Read command line options."""
    parser = argparse.ArgumentParser(
        description="Load the CT series recorded in each Session.mat."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--session-manifest",
        type=Path,
        default=DEFAULT_SESSION_MANIFEST,
    )
    parser.add_argument("--patients", nargs="*")
    parser.add_argument("--report-name", default="matching_report")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Verify the selected series without loading pixel data.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-qc", action="store_true")
    parser.add_argument("--compressed", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the CT and mask preparation pipeline."""
    args = parse_args()
    manifest = load_session_manifest(args.session_manifest)
    patients = (
        sorted(args.patients, key=patient_sort_key)
        if args.patients
        else sorted(manifest, key=patient_sort_key)
    )
    missing = [patient_id for patient_id in patients if patient_id not in manifest]
    if missing:
        raise ValueError(f"Patients missing from manifest: {', '.join(missing)}")

    print(f"Data root: {args.data_root.resolve()}")
    print(f"Output root: {args.output_root.resolve()}")
    print(f"Session manifest: {args.session_manifest.resolve()}")
    print(f"Patients: {len(patients)}")
    print(f"Mode: {'report only' if args.report_only else 'export'}")

    results: list[PatientResult] = []
    for patient_id in tqdm(patients, desc="Patients"):
        try:
            result = process_patient(
                patient_id=patient_id,
                session_series=manifest[patient_id],
                manifest_path=args.session_manifest,
                data_root=args.data_root,
                output_root=args.output_root,
                report_only=args.report_only,
                overwrite=args.overwrite,
                make_qc=not args.no_qc,
                compressed=args.compressed,
            )
        except Exception as error:
            result = PatientResult(
                patient_id=patient_id,
                status="error",
                message=f"{type(error).__name__}: {error}",
                mask_path=None,
                mask_shape_xyz=None,
                selected_series_uid=manifest[patient_id].series_uid,
                selected_series_dir=manifest[patient_id].series_folder,
                ct_shape_zyx=None,
                mask_shape_zyx=None,
                mask_voxels=None,
                mask_volume_ml=None,
            )
        results.append(result)

    write_reports(results, args.output_root, args.report_name)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print(f"Summary: {counts}")
    print(f"Report: {args.output_root / f'{args.report_name}.csv'}")


if __name__ == "__main__":
    main()
