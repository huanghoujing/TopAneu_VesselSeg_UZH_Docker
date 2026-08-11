# TopAneu Vessel Segmentation — Docker

Dockerized inference for TopAneu 36-class cerebral vessel segmentation.
The image contains all Python packages **and** the model weights: an ensemble of
three nnU-Net models (resEncM + plain_conv + primusV3S, `Dataset572_TopAneu_Vessel_36fgCls_wLRSwap`,
fold 4, `checkpoint_final.pth`), plus an optional napari-based screenshot renderer.

## Requirements

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (inference runs on GPU; CPU-only is not supported)
- A recent NVIDIA driver (the image ships PyTorch 2.13 with bundled CUDA)

## Build

From this directory (the build context must contain `nnUNet/`, `run_inference.py`, `Dockerfile`):

```bash
docker build -t topaneu_vesselseg_uzh .
```

The model weights in `nnUNet/data/results/Dataset572_TopAneu_Vessel_36fgCls_wLRSwap/`
(~790 MB, only `checkpoint_final.pth` + plans/dataset JSONs + logs) are baked into the image.

## Run (interactive mode)

```bash
docker run --rm -it \
    --gpus all \
    --ipc=host \
    -v /path/to/your/images:/input \
    -v /path/to/your/results:/output \
    topaneu_vesselseg_uzh
```

This drops you into a bash shell at `/app/nnUNet` with the Python virtualenv
already on `PATH`.

- `--gpus all` — expose GPUs (or e.g. `--gpus '"device=0"'` for a specific one)
- `--ipc=host` — recommended; the inference pipeline uses multiprocessing with shared memory
- `-v ...:/input`, `-v ...:/output` — mount your data; any paths work, `/input` / `/output` are just conventions

## Inference

Inside the container:

```bash
# Single NIfTI file
python run_inference.py -i /input/case_001.nii.gz -o /output

# Folder: all nested *.nii.gz files are found automatically.
# The output directory mirrors the input sub-folder structure, e.g.
#   /input/center1/case_001.nii.gz  ->  /output/center1/case_001.nii.gz
python run_inference.py -i /input -o /output
```

One-shot (non-interactive) usage works too:

```bash
docker run --rm --gpus all --ipc=host \
    -v /path/to/images:/input -v /path/to/results:/output \
    topaneu_vesselseg_uzh \
    python run_inference.py -i /input -o /output
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `-i, --input` | (required) | NIfTI file, or folder searched recursively |
| `-o, --output` | (required) | Output dir; mirrors input sub-folder structure |
| `--suffix` | `.nii.gz` | Input filename suffix to match; stripped from the case name. Use `_0000.nii.gz` for nnU-Net channel-suffixed images (output then drops `_0000`), or `.nii` for uncompressed NIfTI |
| `--output-ext` | `.nii.gz` | Extension of saved segmentations |
| `--sequential` | off | Run the pipeline sequentially in one process instead of parallel workers |
| `--n-infer-workers` | `1` | GPU inference workers |
| `--n-pre-post-workers` | `2` | Preprocessing workers and post-processing workers (also the queue sizes) |
| `--n-gpus` | `1` | GPUs to spread inference workers over |
| `--overwrite-existing` | off | Re-run cases whose output already exists (default: skip them) |
| `--vis` | off | Render napari screenshot galleries of the predictions after inference |
| `--vis-out-dir` | `<output>_VIZ` | Where to save screenshot PNGs (mirrors sub-folder structure) |
| `--vis-views` | `anterior left superior x y z` | Views per gallery |
| `--vis-grid-cols` | `3` | Gallery grid columns |

### CPU thread limits

Thread counts are baked in as environment variables with default `4` and can be
overridden at `docker run` time:

```bash
docker run --rm -it --gpus all --ipc=host \
    -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8 \
    -v /path/to/images:/input -v /path/to/results:/output \
    topaneu_vesselseg_uzh
```

`nnUNet_n_proc_DA=1` is also set (the pipeline does not use the nnU-Net dataloader).

## Screenshot visualization (optional)

Pass `--vis` to `run_inference.py` to render a multi-view 3D gallery PNG per
predicted case (prediction overlaid on the input image), using
`nnunetv2/houjing_scripts/vis_label_screenshots_napari_multi_view.py` under
`xvfb` (no display needed). Note: the renderer only picks up `.nii.gz` label
files, so keep the default `--output-ext .nii.gz` when using `--vis`.

```bash
python run_inference.py -i /input -o /output --vis --vis-out-dir /output_viz
```

For full control (label alpha, canvas size, sampling, …) run the script directly:

```bash
xvfb-run -a python nnunetv2/houjing_scripts/vis_label_screenshots_napari_multi_view.py \
    --labels_dir /output --images_dir /input --out_dir /output_viz \
    --views anterior left superior x y z --grid_cols 3 --skip_existing
```

## Notes

- **Skip-existing is on by default**: re-running the same command only processes
  cases without an existing output file — convenient for resuming interrupted runs.
- **Invalid NIfTI files**: unreadable inputs are logged and skipped; the rest of
  the batch continues. To pre-check a folder:
  `python nnunetv2/houjing_scripts/20260602_topaneu_vessel/D571_D572/inference_demo/check_invalid_nifti.py --input-dir /input --output-file /output/invalid_nifti.txt --read-pixels`
- **Duplicate basenames** in different sub-folders are handled automatically
  (processed in sequential batches to avoid temp-file collisions).
- **Custom weights**: set `-e TOPANEU_MODEL_ROOT=/path/to/weights` to point the
  runner at a mounted weights directory with the same three model sub-folders.
- Temporary preprocessed `.npz` files are written to `<output>/tmp_preprocessed/`
  during a run and cleaned up as cases complete.
