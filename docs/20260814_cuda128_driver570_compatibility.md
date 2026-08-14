# CUDA 12.8 Build for Driver 570 Compatibility (and Why the Image Grew)

*2026-08-14. Applies to the `topaneu_vesselseg_uzh` image (`Dockerfile` +
`nnUNet/pyproject.toml` + `nnUNet/uv.lock`).*

## Problem

The image failed to run on machines with NVIDIA driver r570. The image has no
CUDA base image — `python:3.12-slim-bookworm` plus PyTorch wheels that bundle
the CUDA runtime as `nvidia-*` pip packages. The old `uv.lock` resolved
**torch 2.13.0 from PyPI**, and PyPI torch ≥ 2.12 ships **CUDA 13.0 builds
only** (`nvidia-cuda-runtime 13.0.96` etc.). CUDA 13 requires driver r580+;
driver r570 supports at most CUDA 12.8. Result: `torch.cuda.is_available()`
is False (or CUDA init errors) on r570 hosts.

## Fix

Pin torch and torchvision to CUDA 12.8 builds from PyTorch's own wheel index.
**torch 2.11.0 is the newest release that still publishes cu128 wheels** —
2.12+ are CUDA-13-only everywhere, including `download.pytorch.org`.

Changes in `nnUNet/pyproject.toml`:

- `torch>=2.1.2,<2.12` — keeps the resolver away from CUDA-13-only releases.
- `torchvision` added as a **direct** dependency. It was previously only
  transitive (via `timm`), and `[tool.uv.sources]` pins apply to direct
  dependencies only. Without this, uv kept resolving torchvision from PyPI —
  a CUDA 13 build that would fail to load against cu128 torch.
- Index pin, Linux only (macOS/Windows keep PyPI wheels):

  ```toml
  [tool.uv.sources]
  torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
  torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]

  [[tool.uv.index]]
  name = "pytorch-cu128"
  url = "https://download.pytorch.org/whl/cu128"
  explicit = true
  ```

After `uv lock`: torch 2.11.0+cu128, torchvision 0.26.0+cu128, and every
`nvidia-*` package is from the `-cu12` 12.8.x series (cuDNN 9.19, NCCL 2.28.9,
triton 3.6.0). No `cu13` packages remain in the lock.

Verified in the rebuilt image (RTX PRO 6000 Blackwell host):
`torch 2.11.0+cu128`, `torch.version.cuda == '12.8'`, GPU matmul OK.
The cu128 wheels include sm_120 kernels, so Blackwell GPUs still work.

## Why the image grew from ~6.8 GB to ~9.7 GB

The ~2.9 GB increase is the CUDA 12.8 user-mode libraries themselves — the
direct price of driver-570 compatibility, not extra packages. CUDA 13 cut
binary sizes roughly in half by dropping precompiled kernels for older GPU
architectures (Maxwell/Pascal/Volta) and compressing the remaining
per-architecture fatbins much better. The cu12 builds still carry kernels for
every architecture back to sm_50, so nearly every library is 1.5–2× larger.

Compressed wheel sizes, old lock (CUDA 13) vs new lock (CUDA 12.8), for the
Python-3.12 Linux x86_64 resolution actually installed in the image:

| Component  | CUDA 13 (old) | CUDA 12.8 (new) |
|------------|---------------|-----------------|
| torch      | 527 MB        | ~888 MB         |
| cuDNN      | 366 MB        | 658 MB          |
| cuBLAS     | 423 MB        | 594 MB          |
| cuSPARSE   | 146 MB        | 288 MB          |
| cuSPARSELt | 170 MB        | 287 MB          |
| NCCL       | 206 MB        | 297 MB          |
| nvSHMEM    | 60 MB         | 139 MB          |

Uncompressed inside the image, the GPU stack is now ~6.5 GB
(`torch` 1.6 GB + `nvidia/` 4.3 GB + `triton` 0.6 GB, measured with `du` in
the container); with CUDA 13 the same stack was ~3.5 GB.

There is no smaller official build that runs on driver 570: cu126 wheels only
exist for older torch versions and are not meaningfully smaller, and stripping
unused architectures out of NVIDIA's libraries is not practically supportable.
If distribution size matters, compress the export — the fatbins compress
reasonably well:

```bash
docker save topaneu_vesselseg_uzh | gzip > topaneu_vesselseg_uzh.tar.gz
```

## Output equivalence

Predictions from the cu128/torch-2.11 image were compared voxel-wise against
the reference predictions from the CUDA 13/torch-2.13 image (5 test cases,
`nnunetv2/houjing_scripts/compare_nifti_folders.py`, folders
`data/tmp/20260814_DEBUG_MEM/20260814_test_docker_cuda128_torch211` vs
`data/tmp/20260814_DEBUG_MEM/test_pred_1infer_1pred_vmem16g_mem32g_4thread`):

| Case | Differing voxels | Total voxels | Foreground Dice |
|---|---|---|---|
| topaneu_center1_mr_148 | 12 | 31,500,336 | 0.999954 |
| topaneu_center2_ct_107 | 8  | 20,380,280 | 0.999946 |
| topaneu_center2_mr_031 | 22 | 48,862,200 | 0.999934 |
| topaneu_center4_ct_030 | 43 | 81,185,940 | 0.999925 |
| topaneu_center5_mr_027 | 13 | 31,457,280 | 0.999941 |

Geometry (shape/affine/dtype) matches everywhere; each case differs in at most
43 of tens of millions of voxels (≤ 0.0001 %). The disagreements are scattered
single voxels, almost all label↔background flips at structure boundaries — the
expected signature of tiny floating-point differences (different cuDNN/kernel
selections between the two torch builds shift logits at the ~1e-6 level)
flipping an argmax where two classes are nearly tied. No systematic behavior
change; the cu128 rebuild is output-equivalent for practical purposes.

## When to revert

Once all target machines run driver r580+, drop the `<2.12` constraint, the
`torchvision` direct dependency, and the `[tool.uv.sources]`/`[[tool.uv.index]]`
blocks, then re-lock — PyPI's CUDA 13 builds are ~3 GB smaller and track newer
torch releases.
