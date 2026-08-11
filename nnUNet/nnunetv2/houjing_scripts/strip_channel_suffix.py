#!/usr/bin/env python3
"""Strip the nnU-Net channel suffix from image filenames.

Renames every ``*_0000.nii.gz`` in a folder to ``*.nii.gz``, e.g.
``topaneu_center4_ct_002_0000.nii.gz`` -> ``topaneu_center4_ct_002.nii.gz``.

Usage:
    python strip_channel_suffix.py /path/to/folder              # rename
    python strip_channel_suffix.py /path/to/folder --dry-run    # preview only
"""
from __future__ import annotations

import argparse
from pathlib import Path


def strip_channel_suffix(
    folder: str | Path,
    channel: str = "0000",
    *,
    recursive: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> list[tuple[Path, Path]]:
    """Rename ``*_<channel>.nii.gz`` files in *folder* to ``*.nii.gz``.

    Parameters
    ----------
    folder:
        Directory containing the files.
    channel:
        Channel id to strip (default ``"0000"`` -> matches ``_0000.nii.gz``).
    recursive:
        Recurse into sub-directories when True.
    dry_run:
        Report what would happen without touching the filesystem.
    overwrite:
        Allow overwriting an existing destination. When False (default) a
        collision is skipped with a warning so no data is silently lost.

    Returns
    -------
    list of ``(src, dst)`` pairs that were renamed (or would be, for dry_run).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    suffix = f"_{channel}.nii.gz"
    files = folder.rglob(f"*{suffix}") if recursive else folder.glob(f"*{suffix}")

    renamed: list[tuple[Path, Path]] = []
    for src in sorted(files):
        dst = src.with_name(src.name.removesuffix(suffix) + ".nii.gz")
        if dst.exists() and not overwrite:
            print(f"SKIP (target exists): {src.name} -> {dst.name}")
            continue
        if not dry_run:
            src.rename(dst)
        renamed.append((src, dst))
        print(f"{'DRY-RUN' if dry_run else 'RENAMED'}: {src.name} -> {dst.name}")

    verb = "would be " if dry_run else ""
    print(f"\n{len(renamed)} file(s) {verb}renamed.")
    return renamed


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rename *_0000.nii.gz -> *.nii.gz in a folder."
    )
    p.add_argument("folder", type=Path, help="Folder containing the .nii.gz files")
    p.add_argument("--channel", default="0000", help="Channel id to strip (default: 0000)")
    p.add_argument("--recursive", action="store_true", help="Recurse into sub-folders")
    p.add_argument("--dry-run", action="store_true", help="Preview without renaming")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing targets")
    args = p.parse_args()
    strip_channel_suffix(
        args.folder,
        channel=args.channel,
        recursive=args.recursive,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
