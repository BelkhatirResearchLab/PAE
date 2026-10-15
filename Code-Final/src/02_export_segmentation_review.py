"""Export complete axial CT-mask overlays for clinician review.

Uncompressed NPZ archives are memory-mapped. Compressed archives are loaded
normally, and interrupted exports resume by skipping existing PNG files.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import struct
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_MATCHED_ROOT = PROJECT_ROOT / "outputs_final" / "ct_mask_matched"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs_final" / "segmentation_clinical_review"
)
WINDOW_MIN_HU = -200.0
WINDOW_MAX_HU = 400.0
CONTOUR_INDEX = 255
PATIENT_PATTERN = re.compile(r"PAE(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export complete axial CT-mask overlays and HTML viewers."
    )
    parser.add_argument("--matched-root", type=Path, default=DEFAULT_MATCHED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        metavar="PATIENT_ID",
        help=(
            "Export only the listed patients. By default, all PAE patient "
            "folders in --matched-root are exported."
        ),
    )
    parser.add_argument("--window-min", type=float, default=WINDOW_MIN_HU)
    parser.add_argument("--window-max", type=float, default=WINDOW_MAX_HU)
    parser.add_argument("--contour-width", type=int, default=2)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite PNG files that already exist. The default is resume mode.",
    )
    return parser.parse_args()


def patient_sort_key(patient_id: str) -> tuple[int, str]:
    """Sort PAE identifiers by their numeric component."""
    match = PATIENT_PATTERN.fullmatch(patient_id)
    if match is None:
        return (10**9, patient_id)
    return (int(match.group(1)), patient_id)


def resolve_patients(matched_root: Path, requested: list[str] | None) -> list[str]:
    """Return requested patients or discover every PAE patient folder."""
    if not matched_root.is_dir():
        raise FileNotFoundError(f"Matched CT-mask root does not exist: {matched_root}")

    if requested:
        patients = [patient_id.strip().upper() for patient_id in requested]
    else:
        patients = [
            path.name
            for path in matched_root.iterdir()
            if path.is_dir() and PATIENT_PATTERN.fullmatch(path.name)
        ]

    patients = sorted(patients, key=patient_sort_key)
    if not patients:
        raise ValueError(f"No PAE patient folders found in {matched_root}")
    if len(patients) != len(set(patients)):
        raise ValueError("Duplicate patient IDs were requested")

    invalid_ids = [
        patient_id
        for patient_id in patients
        if PATIENT_PATTERN.fullmatch(patient_id) is None
    ]
    if invalid_ids:
        raise ValueError(f"Invalid patient IDs: {', '.join(invalid_ids)}")
    return patients


def open_stored_npy_memmap(npz_path: Path, member_name: str) -> np.memmap:
    """Open an uncompressed NPY member inside an NPZ archive as a memory map."""
    with zipfile.ZipFile(npz_path) as archive:
        info = archive.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"{npz_path.name}:{member_name} is compressed and cannot be "
                "memory-mapped in place"
            )

    with npz_path.open("rb") as stream:
        stream.seek(info.header_offset)
        local_header = stream.read(30)
        if len(local_header) != 30:
            raise ValueError(f"Invalid ZIP local header in {npz_path}")
        signature, = struct.unpack("<I", local_header[:4])
        if signature != 0x04034B50:
            raise ValueError(f"Invalid ZIP signature in {npz_path}")
        filename_length, extra_length = struct.unpack("<HH", local_header[26:30])
        npy_offset = info.header_offset + 30 + filename_length + extra_length
        stream.seek(npy_offset)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream
            )
        elif version in {(2, 0), (3, 0)}:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                stream
            )
        else:
            raise ValueError(f"Unsupported NPY version {version} in {npz_path}")
        array_offset = stream.tell()

    order = "F" if fortran_order else "C"
    return np.memmap(
        npz_path,
        dtype=dtype,
        mode="r",
        offset=array_offset,
        shape=shape,
        order=order,
    )


def load_npz_arrays(npz_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Memory-map stored arrays or load compressed arrays as a fallback."""
    members = ("ct_hu.npy", "mask.npy")
    with zipfile.ZipFile(npz_path) as archive:
        stored = all(
            archive.getinfo(member).compress_type == zipfile.ZIP_STORED
            for member in members
        )

    if stored:
        return (
            open_stored_npy_memmap(npz_path, "ct_hu.npy"),
            open_stored_npy_memmap(npz_path, "mask.npy"),
            "memory-map",
        )

    with np.load(npz_path, allow_pickle=False) as archive:
        ct_hu = np.asarray(archive["ct_hu"])
        mask = np.asarray(archive["mask"])
    return ct_hu, mask, "compressed fallback"


def make_palette() -> list[int]:
    """Build an indexed grayscale palette with red reserved for the contour."""
    palette: list[int] = []
    for index in range(255):
        value = round(index * 255 / 254)
        palette.extend((value, value, value))
    palette.extend((255, 0, 0))
    return palette


PALETTE = make_palette()


def mask_boundary(mask: np.ndarray, width: int) -> np.ndarray:
    """Return a thin boundary without covering the interior anatomy."""
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros_like(binary)

    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    interior = (
        binary
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = binary & ~interior

    for _ in range(max(0, width - 1)):
        padded_boundary = np.pad(
            boundary, 1, mode="constant", constant_values=False
        )
        boundary = (
            boundary
            | padded_boundary[:-2, 1:-1]
            | padded_boundary[2:, 1:-1]
            | padded_boundary[1:-1, :-2]
            | padded_boundary[1:-1, 2:]
        )
    return boundary


def render_slice(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    output_path: Path,
    window_min: float,
    window_max: float,
    contour_width: int,
) -> int:
    """Save one indexed PNG and return its mask voxel count."""
    scaled = np.clip(
        (np.asarray(ct_slice, dtype=np.float32) - window_min)
        / (window_max - window_min),
        0.0,
        1.0,
    )
    indexed = np.rint(scaled * 254).astype(np.uint8)
    binary_mask = np.asarray(mask_slice, dtype=bool)
    voxel_count = int(np.count_nonzero(binary_mask))
    if voxel_count:
        indexed[mask_boundary(binary_mask, contour_width)] = CONTOUR_INDEX

    image = Image.fromarray(indexed)
    image.putpalette(PALETTE)
    image.save(output_path, format="PNG", compress_level=4, optimize=False)
    return voxel_count


def write_patient_viewer(
    output_path: Path,
    patient_id: str,
    n_slices: int,
    mask_slices: list[int],
    max_mask_slice: int,
    mask_volume_ml: float,
) -> None:
    """Write a self-contained keyboard and slider-based local viewer."""
    mask_json = json.dumps(mask_slices)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(patient_id)} segmentation review</title>
<style>
:root {{ color-scheme: dark; font-family: Arial, sans-serif; }}
body {{ margin: 0; background: #101820; color: #fff; }}
header {{ background: #002d4c; padding: 12px 20px; }}
h1 {{ margin: 0 0 4px; font-size: 22px; }}
.meta {{ color: #dbe6ec; font-size: 14px; }}
main {{ display: grid; grid-template-columns: minmax(520px, 1fr) 280px;
       gap: 18px; max-width: 1180px; margin: 16px auto; padding: 0 16px; }}
.image-panel {{ text-align: center; }}
#slice-image {{ width: min(76vh, 100%); max-height: 76vh; object-fit: contain;
                background: #000; border: 1px solid #536a78; }}
.controls, .review {{ background: #182731; border: 1px solid #385264;
                     border-radius: 8px; padding: 14px; }}
button {{ background: #006b8f; color: white; border: 0; border-radius: 5px;
         padding: 9px 12px; margin: 3px; cursor: pointer; font-weight: bold; }}
button:hover {{ background: #0087ad; }}
input[type=range] {{ width: 100%; }}
.slice-label {{ font-size: 18px; font-weight: bold; margin: 8px 0; }}
.mask-present {{ color: #ff6572; }}
textarea {{ width: 100%; min-height: 105px; background: #0f1c24; color: white;
           border: 1px solid #536a78; border-radius: 4px; }}
ul {{ padding-left: 20px; line-height: 1.45; }}
@media (max-width: 850px) {{ main {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(patient_id)} - axial CT-mask segmentation review</h1>
  <div class="meta">{n_slices} slices | mask volume {mask_volume_ml:.1f} mL |
  red contour = prostate mask | CT window -200 to 400 HU</div>
</header>
<main>
  <section class="image-panel">
    <img id="slice-image" alt="Axial CT-mask overlay">
    <div id="slice-label" class="slice-label"></div>
    <input id="slider" type="range" min="0" max="{n_slices - 1}" value="{max_mask_slice}">
    <div>
      <button onclick="step(-1)">Previous</button>
      <button onclick="step(1)">Next</button>
      <button onclick="jump(firstMask)">First mask</button>
      <button onclick="jump(maxMask)">Largest mask</button>
      <button onclick="jump(lastMask)">Last mask</button>
    </div>
    <p>Use Left/Right arrows or the slider. Page Up/Down moves by 10 slices.</p>
  </section>
  <aside>
    <section class="controls">
      <strong>Clinical review checklist</strong>
      <ul>
        <li>Coverage of prostate apex and base</li>
        <li>Boundary follows the prostate</li>
        <li>No bladder, rectum, bone or surrounding tissue included</li>
        <li>Continuity across adjacent slices</li>
        <li>No abrupt jumps, leakage or missing regions</li>
      </ul>
    </section>
    <section class="review" style="margin-top:12px">
      <strong>Reviewer notes</strong>
      <p><label><input type="radio" name="decision"> Accept</label><br>
      <label><input type="radio" name="decision"> Revise</label></p>
      <textarea placeholder="Record slice numbers and comments here..."></textarea>
      <p style="font-size:12px;color:#bdcbd4">Notes remain in this browser session only;
      copy them into the review form before closing.</p>
    </section>
  </aside>
</main>
<script>
const total = {n_slices};
const maskSlices = {mask_json};
const maskSet = new Set(maskSlices);
const firstMask = maskSlices[0];
const lastMask = maskSlices[maskSlices.length - 1];
const maxMask = {max_mask_slice};
let current = maxMask;
const image = document.getElementById('slice-image');
const label = document.getElementById('slice-label');
const slider = document.getElementById('slider');
function filename(index) {{ return `slices/slice_${{String(index).padStart(4, '0')}}.png`; }}
function show(index) {{
  current = Math.max(0, Math.min(total - 1, Number(index)));
  image.src = filename(current);
  slider.value = current;
  const present = maskSet.has(current);
  label.innerHTML = `Slice ${{current + 1}} / ${{total}} (index ${{current}}) - ` +
    (present ? '<span class="mask-present">mask present</span>' : 'no mask');
}}
function step(delta) {{ show(current + delta); }}
function jump(index) {{ show(index); }}
slider.addEventListener('input', event => show(event.target.value));
document.addEventListener('keydown', event => {{
  if (event.key === 'ArrowLeft') step(-1);
  if (event.key === 'ArrowRight') step(1);
  if (event.key === 'PageUp') step(-10);
  if (event.key === 'PageDown') step(10);
}});
show(current);
</script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def create_contact_sheet(
    patient_dir: Path,
    patient_id: str,
    selected_slices: list[int],
) -> Path:
    """Create a small QC contact sheet from clinically useful slices."""
    images = [
        Image.open(patient_dir / "slices" / f"slice_{index:04d}.png").convert("RGB")
        for index in selected_slices
    ]
    tile_size = 320
    title_height = 34
    canvas = Image.new("RGB", (tile_size * len(images), tile_size + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, (index, image) in enumerate(zip(selected_slices, images)):
        image.thumbnail((tile_size, tile_size))
        left = column * tile_size + (tile_size - image.width) // 2
        canvas.paste(image, (left, title_height))
        draw.text(
            (column * tile_size + 10, 10),
            f"{patient_id} | slice {index + 1} (index {index})",
            fill=(0, 45, 76),
            font=font,
        )
        image.close()
    output_path = patient_dir / "review_contact_sheet.png"
    canvas.save(output_path, compress_level=4)
    canvas.close()
    return output_path


def export_patient(
    patient_id: str,
    matched_root: Path,
    output_root: Path,
    window_min: float,
    window_max: float,
    contour_width: int,
    overwrite: bool,
) -> dict[str, object]:
    """Export one patient and return a manifest row."""
    start_time = time.time()
    source_dir = matched_root / patient_id
    npz_path = source_dir / "data.npz"
    metadata_path = source_dir / "metadata.json"
    if not npz_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing matched CT-mask output for {patient_id}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ct_hu, mask, read_mode = load_npz_arrays(npz_path)
    if ct_hu.shape != mask.shape or ct_hu.ndim != 3:
        raise ValueError(f"Invalid shapes for {patient_id}: CT={ct_hu.shape}, mask={mask.shape}")

    patient_dir = output_root / patient_id
    slices_dir = patient_dir / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    mask_counts: list[int] = []
    written = 0
    skipped = 0

    print(
        f"[{patient_id}] exporting {ct_hu.shape[0]} axial slices "
        f"({read_mode})",
        flush=True,
    )
    for index in range(ct_hu.shape[0]):
        output_path = slices_dir / f"slice_{index:04d}.png"
        mask_count = int(np.count_nonzero(mask[index]))
        mask_counts.append(mask_count)
        if output_path.exists() and not overwrite:
            skipped += 1
        else:
            render_slice(
                ct_hu[index],
                mask[index],
                output_path,
                window_min,
                window_max,
                contour_width,
            )
            written += 1
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == ct_hu.shape[0]:
            print(
                f"[{patient_id}] {index + 1}/{ct_hu.shape[0]} "
                f"(written={written}, resumed={skipped})",
                flush=True,
            )

    mask_slices = [index for index, count in enumerate(mask_counts) if count > 0]
    if not mask_slices:
        raise ValueError(f"Mask is empty for {patient_id}")
    max_mask_slice = int(np.argmax(mask_counts))
    first_mask_slice = mask_slices[0]
    last_mask_slice = mask_slices[-1]

    write_patient_viewer(
        patient_dir / "viewer.html",
        patient_id,
        int(ct_hu.shape[0]),
        mask_slices,
        max_mask_slice,
        float(metadata["mask_volume_ml"]),
    )
    create_contact_sheet(
        patient_dir,
        patient_id,
        [first_mask_slice, max_mask_slice, last_mask_slice],
    )

    summary = {
        "patient_id": patient_id,
        "n_slices": int(ct_hu.shape[0]),
        "image_shape_yx": [int(ct_hu.shape[1]), int(ct_hu.shape[2])],
        "first_mask_slice_index": first_mask_slice,
        "max_mask_slice_index": max_mask_slice,
        "last_mask_slice_index": last_mask_slice,
        "n_mask_slices": len(mask_slices),
        "mask_volume_ml": float(metadata["mask_volume_ml"]),
        "window_hu": [window_min, window_max],
        "contour_colour": "red",
        "contour_width_pixels": contour_width,
        "source_npz": str(npz_path),
        "npz_read_mode": read_mode,
        "mask_alignment": metadata.get("mask_alignment", {}),
        "written_png": written,
        "resumed_png": skipped,
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    (patient_dir / "review_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    del ct_hu
    del mask
    return summary


def write_root_index(output_root: Path, summaries: list[dict[str, object]]) -> None:
    """Write a landing page linking every exported patient viewer."""
    cards = []
    for summary in summaries:
        patient_id = str(summary["patient_id"])
        cards.append(
            f"""
<article>
  <img src="{patient_id}/review_contact_sheet.png" alt="{patient_id} contact sheet">
  <h2>{patient_id}</h2>
  <p>{summary['n_slices']} CT slices | {summary['n_mask_slices']} mask slices |
     mask volume {summary['mask_volume_ml']:.1f} mL</p>
  <a href="{patient_id}/viewer.html">Open slice-by-slice viewer</a>
</article>"""
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PAE segmentation clinical review</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; color: #23313b; background: #eef2f3; }}
header {{ background: #002d4c; color: white; padding: 22px 5%; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
       gap: 18px; padding: 24px 5%; }}
article {{ background: white; border-radius: 9px; padding: 15px;
          box-shadow: 0 2px 8px #0002; }}
img {{ width: 100%; border: 1px solid #ccd7dd; }}
a {{ display: inline-block; background: #006b8f; color: white; padding: 10px 14px;
     border-radius: 5px; text-decoration: none; font-weight: bold; }}
.note {{ margin: 0 5% 24px; background: #fff1df; border-left: 5px solid #ca6503;
        padding: 13px; }}
</style></head><body>
<header><h1>PAE CT-mask segmentation clinical review</h1>
<p>{len(summaries)} patients | full axial slice sequences | red prostate contours</p></header>
<main>{''.join(cards)}</main>
<div class="note">Start with each contact sheet, then open the interactive viewer.
Use the keyboard arrows to inspect every slice and record the decision in
<strong>clinical_review_form.csv</strong>.</div>
</body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def verify_export(output_root: Path, summaries: list[dict[str, object]]) -> None:
    """Verify file counts and clinically important representative slices."""
    for summary in summaries:
        patient_id = str(summary["patient_id"])
        patient_dir = output_root / patient_id
        png_files = sorted((patient_dir / "slices").glob("slice_*.png"))
        expected = int(summary["n_slices"])
        if len(png_files) != expected:
            raise ValueError(
                f"{patient_id}: expected {expected} PNG files, found {len(png_files)}"
            )
        sample_indices = {
            0,
            int(summary["first_mask_slice_index"]),
            int(summary["max_mask_slice_index"]),
            int(summary["last_mask_slice_index"]),
            expected - 1,
        }
        for index in sorted(sample_indices):
            path = patient_dir / "slices" / f"slice_{index:04d}.png"
            with Image.open(path) as image:
                image.verify()
        if not (patient_dir / "viewer.html").is_file():
            raise FileNotFoundError(f"Missing viewer for {patient_id}")


def write_manifest(output_root: Path, summaries: list[dict[str, object]]) -> None:
    fields = [
        "patient_id",
        "n_slices",
        "n_mask_slices",
        "first_mask_slice_index",
        "max_mask_slice_index",
        "last_mask_slice_index",
        "mask_volume_ml",
        "written_png",
        "resumed_png",
        "elapsed_seconds",
    ]
    with (output_root / "review_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)


def write_review_form(
    output_root: Path,
    summaries: list[dict[str, object]],
) -> None:
    """Create the clinician review form without discarding existing decisions."""
    path = output_root / "clinical_review_form.csv"
    fields = [
        "patient_id",
        "review_status",
        "reviewer",
        "review_date",
        "slices_for_revision",
        "comments",
    ]
    existing: dict[str, dict[str, str]] = {}
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                patient_id = str(row.get("patient_id", "")).strip().upper()
                if patient_id:
                    existing[patient_id] = {
                        field: str(row.get(field, "")) for field in fields
                    }

    rows = []
    for summary in summaries:
        patient_id = str(summary["patient_id"])
        row = {field: "" for field in fields}
        row["patient_id"] = patient_id
        if patient_id in existing:
            row.update(existing[patient_id])
            row["patient_id"] = patient_id
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_failures(output_root: Path, failures: list[dict[str, str]]) -> None:
    """Write a machine-readable list of patients that could not be exported."""
    path = output_root / "export_failures.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["patient_id", "error"])
        writer.writeheader()
        writer.writerows(failures)


def main() -> None:
    args = parse_args()
    if args.window_max <= args.window_min:
        raise ValueError("--window-max must be greater than --window-min")
    if args.contour_width < 1:
        raise ValueError("--contour-width must be at least 1")

    patients = resolve_patients(args.matched_root, args.patients)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    print(
        f"Preparing clinical review export for {len(patients)} patients",
        flush=True,
    )
    for number, patient_id in enumerate(patients, 1):
        print(f"=== Patient {number}/{len(patients)}: {patient_id} ===", flush=True)
        try:
            summaries.append(
                export_patient(
                    patient_id,
                    args.matched_root,
                    args.output_root,
                    args.window_min,
                    args.window_max,
                    args.contour_width,
                    args.overwrite,
                )
            )
        except Exception as error:  # Continue so one patient does not block all others.
            message = f"{type(error).__name__}: {error}"
            failures.append({"patient_id": patient_id, "error": message})
            print(f"[{patient_id}] ERROR: {message}", flush=True)

    write_manifest(args.output_root, summaries)
    write_root_index(args.output_root, summaries)
    write_review_form(args.output_root, summaries)
    write_failures(args.output_root, failures)
    verify_export(args.output_root, summaries)
    print(
        f"Clinical review export complete: {len(summaries)}/{len(patients)} patients; "
        f"failures={len(failures)}; output={args.output_root}",
        flush=True,
    )
    if failures:
        raise RuntimeError(
            f"Clinical review export had {len(failures)} patient failures; "
            f"see {args.output_root / 'export_failures.csv'}"
        )


if __name__ == "__main__":
    main()
