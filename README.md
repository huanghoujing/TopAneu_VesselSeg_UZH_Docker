# TopAneu Vessel Segmentation — Docker

Dockerized inference for TopAneu 36-class cerebral vessel segmentation.
The image contains all Python packages **and** the model weights: an ensemble of
three nnU-Net models (resEncM + plain_conv + primusV3S, `Dataset572_TopAneu_Vessel_36fgCls_wLRSwap`,
fold 4, `checkpoint_final.pth`), plus an optional napari-based screenshot renderer.

## Requirements

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (inference runs on GPU; CPU-only is not supported)
- A recent NVIDIA driver (the image ships PyTorch 2.13 with bundled CUDA)
- 16 GB of GPU VRAM recommended (the default configuration was validated with
  GPU memory capped to 16 GB; observed peak was ~7 GB in limited tested cases)

## Build

From this directory (the build context must contain `nnUNet/`, `run_inference.py`, `Dockerfile`):

```bash
docker build \
    --build-arg USER_UID=$(id -u) \
    --build-arg USER_GID=$(id -g) \
    -t topaneu_vesselseg_uzh .
```

The model weights in `nnUNet/data/results/Dataset572_TopAneu_Vessel_36fgCls_wLRSwap/`
(~790 MB, only `checkpoint_final.pth` + plans/dataset JSONs + logs) are baked into the image.

The `USER_UID`/`USER_GID` build args make the container user match your host
user (see [File permissions & sudo](#file-permissions--sudo)); if omitted they
default to `1000:1000`. Changing them only rebuilds the final (tiny) image
layer, not the dependency layers.

## Run (interactive mode)

```bash
docker run --rm -it \
    --gpus all \
    --ipc=host \
    --user root \
    --memory=32g --shm-size=32g \
    -v /path/to/your/images:/input \
    -v /path/to/your/results:/output \
    topaneu_vesselseg_uzh
```

This drops you into a bash shell at `/app/nnUNet` with the Python virtualenv
already on `PATH`.

- `--gpus all` — expose GPUs (or e.g. `--gpus '"device=0"'` for a specific one)
- `--ipc=host` — recommended; the inference pipeline uses multiprocessing with shared memory
- `--user root` — required on rootless Docker (see [File permissions & sudo](#file-permissions--sudo));
  on rootful Docker you can drop it to run as the built-in `topaneu` user
- `--memory=32g --shm-size=32g` — validated for the default (parallel) inference
  configuration (the ensemble export streams in slabs with a float16
  accumulator; observed peak was ~19 GB on the largest debug cases). More memory
  only helps when [scaling up workers](#scaling-up-advanced)
- `-v ...:/input`, `-v ...:/output` — mount your data; any paths work, `/input` / `/output` are just conventions

## File permissions & sudo

The container runs as non-root user `topaneu` with **passwordless sudo**
(`sudo apt-get install ...` etc. works inside the container).

**Rootless Docker: add `--user root` to every `docker run`.** Rootless mode
remaps UIDs: *your host user* becomes root inside the container, while the
image's `topaneu` (UID 1000) becomes an unprivileged subuid that cannot write
to your mounted files (typically a `PermissionError` / `Permission denied` on
the output dir, while reading still works). Running as container root is safe
in rootless mode — it is just your own host user, and output files end up
owned by you. The `USER_UID`/`USER_GID` build args and the advice below apply
to regular (rootful) Docker.

Mounted host files keep their host owner (UID/GID) inside the container. So:

- **Build with `--build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)`**
  (as shown above). The container user then has the *same UID/GID as you*:
  it can read your mounted input files, and everything it writes to `/output`
  is owned by you on the host — no `root`-owned result files to clean up.
- If you got an image built for a *different* UID (e.g. from someone else),
  override the user at runtime instead of rebuilding:

  ```bash
  docker run --rm -it --gpus all --ipc=host \
      --user $(id -u):$(id -g) -e HOME=/tmp \
      -v /path/to/images:/input -v /path/to/results:/output \
      topaneu_vesselseg_uzh
  ```

  (`-e HOME=/tmp` gives config files a writable home; `sudo` is not available
  in this mode since the anonymous UID is not in the sudoers file.)
- If your input data is only readable by another group, add the group with
  `--group-add <gid>`.
- **Create the output directory yourself before `docker run`** (e.g.
  `mkdir -p /path/to/results`). If a bind-mount path does not exist, the Docker
  daemon auto-creates it **owned by root**, and the container user then gets
  `PermissionError: [Errno 13]` when writing to it. If that already happened,
  fix it with `sudo chown $(id -u):$(id -g) /path/to/results`.

## Inference

Inside the container (default configuration — the parallel pipeline with one
preprocessing, one GPU, and one post-processing worker):

```bash
# Single NIfTI file
python run_inference.py -i /input/case_001.nii.gz -o /output

# Folder: all nested *.nii.gz files are found automatically.
# The output directory mirrors the input sub-folder structure, e.g.
#   /input/center1/case_001.nii.gz  ->  /output/center1/case_001.nii.gz
python run_inference.py -i /input -o /output
```

End-to-end (non-interactive) usage:

```bash
docker run --rm --gpus all --ipc=host \
    --user root \
    --memory=32g --shm-size=32g \
    -v /path/to/images:/input -v /path/to/results:/output \
    topaneu_vesselseg_uzh \
    python run_inference.py -i /input -o /output --gpu_limit_GB 16
```

This configuration was validated end-to-end on 5 example cases (mixed MR/CT
cases from all centers): ~100 s/case
on a modern GPU under concurrent host load (timing varies with host load; ~45
s/case was measured on an idle machine before the memory optimizations),
GPU memory capped to 16 GB (observed VRAM peak ~7 GB), peak container memory
19.3 GB — comfortably inside `--memory=32g`.

The analysis and benchmarks behind the memory/speed defaults (slab-streamed
ensemble export, fp16 accumulator, `--fuse_logits` fusion trade-offs) are
documented in
[docs/20260814_ensemble_export_optimization.md](docs/20260814_ensemble_export_optimization.md).

### Options

| Option | Default | Meaning |
|---|---|---|
| `-i, --input` | (required) | NIfTI file, or folder searched recursively |
| `-o, --output` | (required) | Output dir; mirrors input sub-folder structure |
| `--suffix` | `.nii.gz` | Input filename suffix to match; stripped from the case name. Use `_0000.nii.gz` for nnU-Net channel-suffixed images (output then drops `_0000`), or `.nii` for uncompressed NIfTI |
| `--output_ext` | `.nii.gz` | Extension of saved segmentations |
| `--sequential` | off | Run the pipeline sequentially in one process instead of parallel workers |
| `--n_infer_workers` | `1` | GPU inference workers |
| `--n_pre_post_workers` | `1` | Preprocessing workers and post-processing workers (also the queue sizes); more is faster but holds more cases in RAM |
| `--n_gpus` | `1` | GPUs to spread inference workers over |
| `--gpu_limit_GB` | none | Approximate GPU memory cap in GB, applied in each inference process (works in both sequential and parallel mode) |
| `--fp16` / `--no-fp16` | on | Keep the ensemble probability accumulator in float16 (~halves its RAM). `--no-fp16` restores the float32 accumulator; outputs differ only at near-exact probability ties, below run-to-run GPU nondeterminism |
| `--fuse_logits` | off | Average model logits on the preprocessed grid and resample once instead of per model (~2x faster inference+export stage, lower RAM). Changes fusion from mean-of-softmax to softmax-of-mean-logits: a slightly more conservative segmentation, differing at low-confidence structure boundaries (~1.6% of foreground voxels) and dropping tiny uncertain structures (Pcom/AICA scale) slightly more often. Only valid when all ensemble models share the same preprocessing target spacing — true for the built-in 3-model ensemble, enforced at runtime (relevant with `TOPANEU_MODEL_ROOT` custom weights) |
| `--overwrite_existing` | off | Re-run cases whose output already exists (default: skip them) |
| `--vis` | off | Render napari screenshot galleries of the predictions after inference |
| `--vis_out_dir` | `<output>/viz` | Where to save screenshot PNGs (mirrors sub-folder structure) |
| `--vis_views` | `anterior left superior x y z` | Views per gallery |
| `--vis_grid_cols` | `3` | Gallery grid columns |

### CPU thread limits

Thread counts (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`)
are baked into the image with default `4`, which the default configuration was
validated with. Override them at `docker run` time with `-e OMP_NUM_THREADS=1
-e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1`
etc. — lower values reduce RAM pressure at the cost of slower pre/post
processing (with threads at 1, inference took ~25% longer in our tests).
`nnUNet_n_proc_DA=1` is also set (the pipeline does not use the nnU-Net
dataloader).

### Low-VRAM fallback (sequential)

The default (parallel) pipeline fits in `--memory=32g` (see above). If your
GPU has less than 16 GB of VRAM, run everything in a single process with
reduced thread counts — but it is much slower:

```bash
docker run --rm --gpus all --ipc=host \
    --user root \
    --memory=32g --shm-size=32g \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -v /path/to/images:/input -v /path/to/results:/output \
    topaneu_vesselseg_uzh \
    python run_inference.py -i /input -o /output --sequential --gpu_limit_GB 16
```

### Scaling up (advanced)

- `--n_pre_post_workers 2` (or more) overlaps more preprocessing/postprocessing
  with GPU work — each extra worker holds additional cases in RAM, so raise
  `--memory` accordingly.
- `--n_gpus N` with `--n_infer_workers N` spreads inference workers over
  multiple GPUs.

### Strip Channel Sub-String from Prediction Output Filename

- `--suffix _0000.nii.gz` strips the nnU-Net channel suffix from output names
  (`case_0000.nii.gz` -> `case.nii.gz`).

## Screenshot visualization (optional)

Pass `--vis` to `run_inference.py` to render a multi-view 3D gallery PNG per
predicted case (prediction overlaid on the input image), using
`nnunetv2/houjing_scripts/vis_label_screenshots_napari_multi_view.py` under
`xvfb` (no display needed). Note: the renderer only picks up `.nii.gz` label
files, so keep the default `--output_ext .nii.gz` when using `--vis`.

```bash
python run_inference.py -i /input -o /output --vis
```

Screenshots go to `<output>/viz` by default, i.e. **inside the mounted output
volume**, so they are visible on the host. If you pass `--vis_out_dir`, make
sure it points inside a mounted path — a container-only path like `/output_viz`
(a sibling of the `/output` mount) is lost when the container exits.

A progress bar is shown during rendering, with the cumulative disk usage of the
gallery PNGs and the free space on the target disk (also appended to each
per-file log line).

For full control (label alpha, canvas size, sampling, …) run the script
directly. Start `Xvfb` explicitly rather than through `xvfb-run` — as the
container's first process, `xvfb-run` can hang forever waiting for the X
server's readiness signal:

```bash
Xvfb :99 -screen 0 1280x1024x24 &
DISPLAY=:99 python nnunetv2/houjing_scripts/vis_label_screenshots_napari_multi_view.py \
    --labels_dir /output --images_dir /input --out_dir /output/viz \
    --views anterior left superior x y z --grid_cols 3 --skip_existing
```

## Notes

- **Skip-existing is on by default**: re-running the same command only processes
  cases without an existing output file — convenient for resuming interrupted runs.
- **Invalid NIfTI files**: unreadable inputs are logged and skipped; the rest of
  the batch continues. To pre-check a folder:
  `python nnunetv2/houjing_scripts/20260602_topaneu_vessel/D571_D572/inference_demo/check_invalid_nifti.py --input-dir /input --output-file /output/invalid_nifti.txt --read-pixels`
- **Custom weights**: set `-e TOPANEU_MODEL_ROOT=/path/to/weights` to point the
  runner at a mounted weights directory with the same three model sub-folders.
- **Temporary files**: preprocessed cases are staged as compressed `.npz` files
  in `<output>/tmp_preprocessed/`. Disk usage there is bounded, not proportional
  to dataset size: the hand-off queue is bounded (`--n_pre_post_workers` slots),
  preprocessing blocks when it is full, and the GPU worker deletes each `.npz`
  immediately after loading it — so at most about
  `2 * n_pre_post_workers + n_infer_workers` cases (~5 with defaults) exist at
  any moment. Leftovers are removed and the folder deleted at the end of the run;
  after a crash/kill, stale files there are safe to delete manually. Temp
  filenames include a hash of the relative input path, so files with the same
  basename in different sub-folders cannot collide.

## Shipping the image as a tar

The image is fully self-contained (packages + model weights), so a single tar
file is all a recipient needs — no registry, no build, no separate weight files.

On the build machine:

```bash
docker save -o topaneu_vesselseg_uzh.tar topaneu_vesselseg_uzh
```

On the target machine (needs Docker + NVIDIA Container Toolkit + NVIDIA driver):

```bash
# 1. load the image (one-time; the tar can be deleted afterwards)
docker load -i topaneu_vesselseg_uzh.tar

# 2. run inference (default configuration)
#    Optional: also render screenshot galleries (adds --vis; PNGs land in /output/viz)
mkdir -p /path/to/results
docker run --rm --gpus all --ipc=host \
    --user root \
    --memory=32g --shm-size=32g \
    -v /path/to/images:/input -v /path/to/results:/output \
    topaneu_vesselseg_uzh \
    python run_inference.py -i /input -o /output --gpu_limit_GB 16 #--vis

```

Or drop into an interactive shell with the same resources (see
[Run (interactive mode)](#run-interactive-mode)) and call `python
run_inference.py ...` from there.
