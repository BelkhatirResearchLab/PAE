"""Launch the original 3D CERR radiomics extraction in MATLAB."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def matlab_string(value: Path) -> str:
    """Escape a path for a MATLAB single-quoted string."""
    return str(value).replace("'", "''")


def build_command(matlab: str, matlab_script: Path, cerr_path: Path) -> list[str]:
    """Build the non-interactive MATLAB command."""
    batch = (
        f"setenv('CERR_ROOT','{matlab_string(cerr_path)}'); "
        f"run('{matlab_string(matlab_script)}');"
    )
    return [matlab, "-batch", batch]


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    code_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the complete Original 3D CERR feature extraction."
    )
    parser.add_argument(
        "--cerr-path",
        type=Path,
        default=code_root.parent / "external" / "CERR",
        help="Path to the CERR package.",
    )
    parser.add_argument(
        "--matlab",
        default="matlab",
        help="MATLAB executable name or full path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the MATLAB command without running it.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the environment and run the MATLAB extraction."""
    args = parse_args()
    matlab_script = Path(__file__).resolve().with_name(
        "step04_extract_cerr_radiomics.m"
    )
    cerr_path = args.cerr_path.resolve()

    if not matlab_script.is_file():
        raise FileNotFoundError(f"MATLAB extraction script not found: {matlab_script}")
    if not cerr_path.is_dir():
        raise FileNotFoundError(f"CERR directory not found: {cerr_path}")

    command = build_command(args.matlab, matlab_script, cerr_path)
    if args.dry_run:
        print(subprocess.list2cmdline(command))
        return

    if shutil.which(args.matlab) is None:
        raise FileNotFoundError(
            f"MATLAB executable not found: {args.matlab}. "
            "Add MATLAB to PATH or use --matlab."
        )

    # MATLAB performs the actual CERR feature calculations.
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
