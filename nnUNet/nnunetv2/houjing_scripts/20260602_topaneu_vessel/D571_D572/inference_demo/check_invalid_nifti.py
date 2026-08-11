"""
There are errors loading some NIFTI cases:

```
2026-07-07 00:13:41.737 | INFO     | nnunetv2.houjing_scripts.infer_ppl_parallel_npz:preprocess_worker:177 - [preprocess_worker] Error processing data/raw/topaneu_batch2/To-Be-Vesseled/topaneu_center1_mr_842_0000.nii.gz: Exception thrown in SimpleITK ImageFileReader_Execute: /work/src/C
ode/IO/src/sitkImageReaderBase.cxx:99:                                                                                                                                                                                                                                                         
sitk::ERROR: Unable to determine ImageIO reader for "data/raw/topaneu_batch2/To-Be-Vesseled/topaneu_center1_mr_842_0000.nii.gz"                                                                                                                                                                
Traceback (most recent call last):                                                                                                                                                                                                                                                             
  File "/mnt/x/data2/Project/TopCoW_Algo_Submission/task-1-seg/nnUNet_TopCoW/nnunetv2/houjing_scripts/infer_ppl_parallel_npz.py", line 141, in preprocess_worker                                                                               
    image = sitk.ReadImage(input_file)                                                                                                                                                                                                                                                         
            ^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                                                                                                                                                                         
  File "/mnt/x/data2/Project/TopCoW_Algo_Submission/task-1-seg/nnUNet_TopCoW/.venv/lib/python3.12/site-packages/SimpleITK/extra.py", line 384, in ReadImage                                                                                             
    return reader.Execute()                                                                                                                                                                                                                                                                    
           ^^^^^^^^^^^^^^^^                                                                                                                                                                                                                                                                    
  File "/mnt/x/data2/Project/TopCoW_Algo_Submission/task-1-seg/nnUNet_TopCoW/.venv/lib/python3.12/site-packages/SimpleITK/SimpleITK.py", line 8534, in Execute
    return _SimpleITK.ImageFileReader_Execute(self)                                                                                                                                                                                                                                            
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                                                                                                                                                            
RuntimeError: Exception thrown in SimpleITK ImageFileReader_Execute: /work/src/Code/IO/src/sitkImageReaderBase.cxx:99:
sitk::ERROR: Unable to determine ImageIO reader for "data/raw/topaneu_batch2/To-Be-Vesseled/topaneu_center1_mr_842_0000.nii.gz"
```

Need to check how many there are in folder `data/raw/topaneu_batch2/To-Be-Vesseled`, saving their filenames and sizes to `data/results/20260706_infer_topaneu_batch2/invalid_nifti.txt`.

.venv/bin/python nnunetv2/houjing_scripts/20260602_topaneu_vessel/D571_D572/20260706_topaneu_batch2/check_invalid_nifti.py --read-pixels
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUT_DIR = REPO_ROOT / "data/raw/20260708_batch_1_2_updated/Train/images"
DEFAULT_OUTPUT_FILE = REPO_ROOT / "data/results/20260708_infer_topaneu/invalid_nifti.txt"


@dataclass(frozen=True)
class CheckResult:
    path: Path
    size_bytes: int
    error: str | None

    @property
    def is_valid(self) -> bool:
        return self.error is None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find NIfTI files that SimpleITK cannot open and write their "
            "filenames and sizes to a report."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing NIfTI files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Report path. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--read-pixels",
        action="store_true",
        help=(
            "Read the full image payload instead of only checking image metadata. "
            "This is slower but can catch truncated payloads with readable headers."
        ),
    )
    return parser.parse_args()


def iter_nifti_files(input_dir: Path) -> list[Path]:
    nifti_files = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and (path.name.endswith(".nii.gz") or path.name.endswith(".nii"))
    ]
    return sorted(nifti_files, key=lambda path: path.name)


def format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{size_bytes} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    raise RuntimeError("unreachable")


def one_line_error(error: Exception) -> str:
    return " ".join(str(error).split())


def check_nifti(path: Path, read_pixels: bool) -> CheckResult:
    size_bytes = path.stat().st_size
    try:
        if read_pixels:
            sitk.ReadImage(str(path))
        else:
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
            reader.ReadImageInformation()
    except Exception as error:  # SimpleITK raises RuntimeError for unreadable files.
        return CheckResult(path=path, size_bytes=size_bytes, error=one_line_error(error))

    return CheckResult(path=path, size_bytes=size_bytes, error=None)


def write_report(
    results: list[CheckResult],
    input_dir: Path,
    output_file: Path,
    read_pixels: bool,
    elapsed_seconds: float,
) -> None:
    invalid_results = [result for result in results if not result.is_valid]
    validator = "SimpleITK.ReadImage" if read_pixels else "SimpleITK.ImageFileReader.ReadImageInformation"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write("# Invalid NIfTI report\n")
        f.write(f"# input_dir\t{input_dir}\n")
        f.write(f"# validator\t{validator}\n")
        f.write(f"# checked_files\t{len(results)}\n")
        f.write(f"# invalid_files\t{len(invalid_results)}\n")
        f.write(f"# elapsed_seconds\t{elapsed_seconds:.2f}\n")
        f.write("filename\tsize_bytes\tsize_human\terror\n")
        for result in invalid_results:
            f.write(
                f"{result.path.name}\t"
                f"{result.size_bytes}\t"
                f"{format_size(result.size_bytes)}\t"
                f"{result.error}\n"
            )


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    nifti_files = iter_nifti_files(input_dir)
    if not nifti_files:
        print(f"No .nii or .nii.gz files found in {input_dir}", file=sys.stderr)
        return 1

    print(f"Checking {len(nifti_files)} NIfTI files in {input_dir}")
    start_time = time.monotonic()
    results: list[CheckResult] = []

    for index, path in enumerate(nifti_files, start=1):
        result = check_nifti(path, read_pixels=args.read_pixels)
        results.append(result)
        if index == len(nifti_files) or index % 25 == 0:
            invalid_count = sum(not item.is_valid for item in results)
            print(f"Checked {index}/{len(nifti_files)} files; invalid so far: {invalid_count}", flush=True)

    elapsed_seconds = time.monotonic() - start_time
    write_report(
        results=results,
        input_dir=input_dir,
        output_file=output_file,
        read_pixels=args.read_pixels,
        elapsed_seconds=elapsed_seconds,
    )

    invalid_results = [result for result in results if not result.is_valid]
    print(f"Invalid files: {len(invalid_results)} / {len(results)}")
    print(f"Wrote report: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
