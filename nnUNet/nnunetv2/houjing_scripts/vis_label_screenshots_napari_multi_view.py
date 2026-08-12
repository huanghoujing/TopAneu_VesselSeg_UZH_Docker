#!/usr/bin/env python
"""Generate a multi-view 3D napari gallery PNG per NIfTI label file (optionally over CT/MR).

For every ``*.nii.gz`` label file in ``--labels_dir`` this renders the labels in 3D from
several orthogonal camera directions and tiles those views into a single gallery PNG saved
to ``--out_dir``. By default it renders 6 views (3 anatomical axes, each from 2 opposite
directions): anterior / posterior, left / right, superior / inferior. Labels are colored by
a fixed-seed random colormap (consistent across all files). If ``--images_dir`` is given, the
matching CT/MR image is shown faintly behind the labels, and three extra 2D tiles are added
showing the orthogonal middle slices (sagittal / coronal / axial) with the labels alpha-
composited onto the grayscale image (opacity set by ``--label_alpha``).

The script must run under a virtual framebuffer on a headless host, e.g.::

    xvfb-run -a ./.venv/bin/python \
        nnunetv2/houjing_scripts/vis_label_screenshots_napari_multi_view.py \
        --labels_dir /path/to/labelsTr \
        --images_dir /path/to/imagesTr \
        --out_dir    /path/to/screenshots \
        --num_samples 5 --shuffle

nnUNet filename convention: label ``X.nii.gz`` matches image ``X_0000.nii.gz`` (channel 0);
an identically named ``X.nii.gz`` in ``--images_dir`` is also accepted.
"""

import argparse
import math
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm
import nibabel as nib
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    io_orientation,
    ornt_transform,
)


def human_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024

# Fallback platform for the rare case xvfb is unavailable; xvfb-run sets DISPLAY and
# overrides this. Must be set before napari/Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TARGET_AXCODES = ("L", "P", "S")  # array axes after load_lps: 0=L/R, 1=A/P, 2=I/S

# Canonical views for LPS-oriented data. Each entry is
#   name -> (view_direction, up_direction)
# where view_direction is the direction the camera looks ALONG, in data-axis order.
VIEWS = {
    "anterior": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),   # see the front
    "posterior": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),  # see the back
    "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),       # see the left side
    "right": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),       # see the right side
    "superior": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),   # see the top
    "inferior": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),    # see the bottom
}
DEFAULT_VIEWS = ["anterior", "posterior", "left", "right", "superior", "inferior"]

# Names of the three orthogonal middle-slice (2D) overlay tiles. Only rendered when an
# image is given (labels are alpha-composited onto the grayscale image slice).
MID_SLICES = ["mid_sagittal", "mid_coronal", "mid_axial"]

# Short axis aliases for the slices (LPS: x=L=sagittal, y=P=coronal, z=S=axial), so a slice
# can be requested as e.g. "z" instead of "mid_axial".
SLICE_ALIASES = {
    "x": "mid_sagittal", "sagittal": "mid_sagittal",
    "y": "mid_coronal", "coronal": "mid_coronal",
    "z": "mid_axial", "axial": "mid_axial",
}

# Every name accepted by --views, and the default selection (all directional views, then the
# three slices which are silently skipped per-file when no image is available).
ALL_VIEW_CHOICES = DEFAULT_VIEWS + MID_SLICES + list(SLICE_ALIASES)
DEFAULT_SELECTION = DEFAULT_VIEWS + MID_SLICES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels_dir", required=True, type=Path, help="folder of *.nii.gz label files")
    p.add_argument("--images_dir", type=Path, default=None, help="folder with matching CT/MR images for overlay")
    p.add_argument("--out_dir", required=True, type=Path, help="output folder for gallery PNGs")
    p.add_argument("--n_labels", type=int, default=60, help="number of label colors in the colormap")
    p.add_argument("--seed", type=int, default=0, help="random seed for the label colormap")
    p.add_argument("--num_samples", type=int, default=None, help="max number of label files to render (default: all)")
    p.add_argument("--shuffle", action="store_true", help="shuffle files before taking --num_samples (random subset)")
    p.add_argument("--sample_seed", type=int, default=0, help="random seed for --shuffle (reproducible sampling)")
    p.add_argument("--views", nargs="+", default=None, choices=ALL_VIEW_CHOICES, metavar="VIEW",
                   help="views to render, in gallery order. 3D directions: "
                        + ", ".join(DEFAULT_VIEWS)
                        + ". Middle slices (need --images_dir): mid_sagittal|x, mid_coronal|y, mid_axial|z. "
                        + "Default: the 6 directions then the 3 slices. "
                        + "E.g. '--views anterior z --grid_cols 2' for one 3D view plus the axial slice.")
    p.add_argument("--grid_cols", type=int, default=2, help="number of columns in the gallery (default 2 -> one axis per row)")
    p.add_argument("--pad", type=int, default=6, help="padding in px between/around gallery tiles")
    p.add_argument("--no_labels", action="store_true", help="do not burn the view name onto each tile")
    p.add_argument("--label_alpha", type=float, default=0.5,
                   help="opacity of labels overlaid on the middle-slice images (0=transparent, 1=opaque)")
    p.add_argument("--size", type=int, nargs=2, default=(512, 512), help="per-view canvas size: W H")
    p.add_argument("--skip_existing", action="store_true", help="skip files whose PNG already exists")
    return p.parse_args()


def build_label_colormap(n_labels, seed):
    """Fixed-seed random colormap.

    Returns ``(cmap, colors)`` where ``cmap`` is the napari DirectLabelColormap and
    ``colors`` is an ``(n_labels + 1, 4)`` RGBA float array (row i = color of label i, with
    row 0 = background). The same ``colors`` are reused for the 2D middle-slice overlays so
    the per-class colors match the 3D views.
    """
    from napari.utils.colormaps import DirectLabelColormap

    rng = np.random.default_rng(seed)
    colors = rng.random((n_labels + 1, 4))
    colors[:, 3] = 1.0  # opaque
    color_dict = {i: colors[i] for i in range(1, n_labels + 1)}
    color_dict[0] = np.array([0.0, 0.0, 0.0, 0.0])  # background transparent
    color_dict[None] = np.array([0.0, 0.0, 0.0, 0.0])  # out-of-range transparent
    return DirectLabelColormap(color_dict=color_dict), colors


def load_lps(nii):
    """Return (array_in_LPS, spacing_in_LPS_axis_order) for a loaded nibabel image."""
    transform = ornt_transform(io_orientation(nii.affine), axcodes2ornt(TARGET_AXCODES))
    arr = apply_orientation(np.asarray(nii.dataobj), transform)
    zooms = np.asarray(nii.header.get_zooms()[:3], dtype=float)
    # transform[:, 0] gives, for each output axis, which input axis it came from.
    spacing = zooms[transform[:, 0].astype(int)]
    return arr, spacing


def find_image(images_dir, label_path):
    """Match a label file to its image trying nnUNet _0000 suffix then identical name."""
    stem = label_path.name[: -len(".nii.gz")]
    for cand in (images_dir / f"{stem}_0000.nii.gz", images_dir / f"{stem}.nii.gz"):
        if cand.exists():
            return cand
    return None


def get_mid_slice(name, vol, spacing):
    """Extract an orthogonal middle 2D slice from an LPS volume (axes 0=L,1=P,2=S).

    Returns ``(slice2d, phys_hw)`` oriented for display (superior up; anterior up for axial),
    where ``phys_hw`` is the physical (height, width) extent used to set the tile aspect ratio.
    """
    if name == "mid_sagittal":  # fix L; plane (P, S) -> rows=S (sup up), cols=P (anterior left)
        s = vol[vol.shape[0] // 2, :, :]
        out = np.flipud(s.T)
        phys = (s.shape[1] * spacing[2], s.shape[0] * spacing[1])
    elif name == "mid_coronal":  # fix P; plane (L, S) -> rows=S (sup up), cols=L
        s = vol[:, vol.shape[1] // 2, :]
        out = np.flipud(s.T)
        phys = (s.shape[1] * spacing[2], s.shape[0] * spacing[0])
    elif name == "mid_axial":  # fix S; plane (L, P) -> rows=P (anterior up), cols=L
        s = vol[:, :, vol.shape[2] // 2]
        out = s.T
        phys = (s.shape[1] * spacing[1], s.shape[0] * spacing[0])
    else:
        raise ValueError(f"unknown mid-slice view: {name}")
    return out, phys


def overlay_slice_tile(img2d, lbl2d, colors, clim, alpha, phys_hw, size):
    """Alpha-composite label colors onto a grayscale image slice and fit into a size tile."""
    from PIL import Image

    n_labels = colors.shape[0] - 1
    lo, hi = clim
    gray = np.clip((img2d.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = np.repeat(gray[..., None], 3, axis=2)

    mask = (lbl2d > 0) & (lbl2d <= n_labels)
    if mask.any():
        lut = colors[:, :3]  # (n_labels+1, 3) in 0..1
        col = lut[np.clip(lbl2d, 0, n_labels)]
        a = alpha * mask[..., None]
        rgb = rgb * (1.0 - a) + col * a

    tile = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    # Resize to fill the tile while preserving the physical aspect ratio; letterbox on black.
    tw, th = size
    ph, pw = phys_hw
    aspect = ph / max(pw, 1e-6)  # physical height / width
    if tw * aspect <= th:
        ow, oh = tw, max(1, int(round(tw * aspect)))
    else:
        oh, ow = th, max(1, int(round(th / aspect)))
    resized = np.asarray(Image.fromarray(tile).resize((ow, oh), Image.NEAREST))
    canvas = np.zeros((th, tw, 3), np.uint8)
    y0, x0 = (th - oh) // 2, (tw - ow) // 2
    canvas[y0 : y0 + oh, x0 : x0 + ow] = resized
    return canvas


def _font(px):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, px)
    return ImageFont.load_default()


def make_gallery(frames, names, cols, pad, label=True):
    """Tile a list of HxWx(3/4) uint8 frames into one RGB gallery image on black."""
    from PIL import Image, ImageDraw

    h, w = frames[0].shape[:2]
    n = len(frames)
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), np.uint8)
    for idx, frame in enumerate(frames):
        r, c = divmod(idx, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y : y + h, x : x + w] = frame[..., :3]

    if label:
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        fnt = _font(max(14, h // 22))
        for idx, name in enumerate(names):
            r, c = divmod(idx, cols)
            y = pad + r * (h + pad)
            x = pad + c * (w + pad)
            draw.text((x + 6, y + 4), name, fill=(255, 255, 0), font=fnt)
        canvas = np.asarray(img)
    return canvas


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.views is None:  # default: 6 directions, plus the 3 slices when an image is given
        views = DEFAULT_VIEWS + (MID_SLICES if args.images_dir is not None else [])
    else:
        views = [SLICE_ALIASES.get(v, v) for v in args.views]  # normalize x/y/z aliases

    label_files = sorted(args.labels_dir.glob("*.nii.gz"))
    if args.shuffle:
        # shuffle a copy of the sorted list so --num_samples picks a random (but
        # reproducible, given --sample_seed) subset instead of the first N.
        np.random.default_rng(args.sample_seed).shuffle(label_files)
    if args.num_samples is not None and args.num_samples >= 0:
        label_files = label_files[: args.num_samples]
    if not label_files:
        print(f"No *.nii.gz files found in {args.labels_dir}")
        return
    print(f"Found {len(label_files)} label file(s); rendering views: {', '.join(views)}")

    import imageio.v3 as iio
    import napari
    from napari._qt.qt_event_loop import get_qapp

    app = get_qapp()
    cmap, colors = build_label_colormap(args.n_labels, args.seed)
    # show=True is required for the canvas to actually render under xvfb (the window lives in
    # the virtual framebuffer). We grab frames via the underlying vispy SceneCanvas.render()
    # (an offscreen FBO render) rather than viewer.screenshot(), because the latter uses Qt's
    # grabFramebuffer() which is flaky/non-deterministic (often black) under software GL.
    viewer = napari.Viewer(show=True, ndisplay=3)
    qt_canvas = viewer.window._qt_viewer.canvas
    scene_canvas = qt_canvas._scene_canvas
    size = tuple(args.size)

    # Pin the on-screen canvas to the offscreen render size. Under xvfb (no window
    # manager) the napari window never reaches its normal size, so the canvas stays
    # tiny; reset_view() then fits the volume to that tiny canvas while render()
    # draws into a bigger FBO — the 3D views come out small and anchored top-left.
    # The canvas widget lives in the window's layout, so it must be fixed at the
    # native Qt level (a plain vispy size assignment is overridden by the layout).
    try:
        scene_canvas.native.setFixedSize(*size)
        for _ in range(5):
            app.processEvents()
        print(f"canvas size pinned to {tuple(scene_canvas.size)} (render size {size})")
    except Exception as e:
        print(f"WARN could not pin canvas size: {e}")

    def grab():
        qt_canvas.on_draw(None)  # bring the scenegraph up to date
        for _ in range(3):
            app.processEvents()
        return scene_canvas.render(size=size, alpha=True)

    # cumulative disk usage of the gallery PNGs (incl. pre-existing ones on resume)
    viz_bytes = sum(f.stat().st_size for f in args.out_dir.glob("*.png"))
    pbar = tqdm(label_files, desc="viz", unit="img", dynamic_ncols=True)
    pbar.set_postfix_str(f"viz={human_bytes(viz_bytes)}, free={human_bytes(shutil.disk_usage(args.out_dir).free)}")
    try:
        for i, label_path in enumerate(pbar, 1):
            out_png = args.out_dir / f"{label_path.name[:-len('.nii.gz')]}.png"
            if args.skip_existing and out_png.exists():
                print(f"[{i}/{len(label_files)}] skip (exists): {out_png.name}")
                continue
            try:
                lbl_arr, spacing = load_lps(nib.load(str(label_path)))
                lbl_arr = lbl_arr.astype(np.int32)

                viewer.layers.clear()

                img_arr = clim = None
                if args.images_dir is not None:
                    image_path = find_image(args.images_dir, label_path)
                    if image_path is None:
                        print(f"[{i}/{len(label_files)}] WARN no matching image for {label_path.name}; labels only")
                    else:
                        img_arr, img_spacing = load_lps(nib.load(str(image_path)))
                        clim = tuple(np.percentile(img_arr, [1, 99]))
                        viewer.add_image(
                            img_arr, scale=img_spacing, contrast_limits=clim,
                            colormap="gray", rendering="mip", opacity=0.5, name="image",
                        )

                viewer.add_labels(lbl_arr, scale=spacing, colormap=cmap, rendering="translucent", name="labels")
                # reset_view centers + zooms to fit at the default orientation; each 3D view
                # then only rotates the camera (keeping that zoom), so all tiles share a scale.
                viewer.reset_view()

                frames, names, skipped = [], [], []
                for name in views:
                    if name in VIEWS:  # 3D directional view rendered by napari
                        view_dir, up_dir = VIEWS[name]
                        viewer.camera.set_view_direction(view_direction=view_dir, up_direction=up_dir)
                        frames.append(grab())
                        names.append(name)
                    else:  # middle-slice overlay; requires the image
                        if img_arr is None:
                            skipped.append(name)
                            continue
                        img2d, phys = get_mid_slice(name, img_arr, spacing)
                        lbl2d, _ = get_mid_slice(name, lbl_arr, spacing)
                        frames.append(overlay_slice_tile(img2d, lbl2d, colors, clim, args.label_alpha, phys, size))
                        names.append(name)

                if skipped:
                    print(f"[{i}/{len(label_files)}] skip slices (no image): {', '.join(skipped)}")
                if not frames:
                    print(f"[{i}/{len(label_files)}] nothing to render for {label_path.name}")
                    continue

                gallery = make_gallery(frames, names, args.grid_cols, args.pad, label=not args.no_labels)
                iio.imwrite(out_png, gallery)
                viz_bytes += out_png.stat().st_size
                free = shutil.disk_usage(args.out_dir).free
                pbar.set_postfix_str(f"viz={human_bytes(viz_bytes)}, free={human_bytes(free)}")
                print(f"[{i}/{len(label_files)}] wrote {out_png.name}  ({len(frames)} views, "
                      f"labels={int(lbl_arr.max())}, viz total={human_bytes(viz_bytes)}, disk free={human_bytes(free)})")
            except Exception as e:  # one bad file should not abort the batch
                print(f"[{i}/{len(label_files)}] ERROR on {label_path.name}: {e}")
    finally:
        pbar.close()
        viewer.close()


if __name__ == "__main__":
    main()
