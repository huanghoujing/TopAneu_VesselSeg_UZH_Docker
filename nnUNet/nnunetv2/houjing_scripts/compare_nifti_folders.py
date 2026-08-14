"""Compare two folders of NIfTI files (e.g. predictions from two builds).

For each file name present in both folders, compares header geometry
(shape, spacing, affine, dtype) and voxel data: exact-match voxel counts,
and for integer label maps also per-label Dice + a confusion summary of the
disagreeing labels. Files present in only one folder are listed.

Usage:
    python compare_nifti_folders.py FOLDER_A FOLDER_B [--dice-per-label]
"""
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def find_niftis(folder: Path):
    return {p.name: p for p in sorted(folder.rglob('*.nii*')) if p.is_file()}


def compare_pair(path_a: Path, path_b: Path, dice_per_label: bool):
    img_a, img_b = nib.load(path_a), nib.load(path_b)
    report = {'name': path_a.name, 'issues': []}

    if img_a.shape != img_b.shape:
        report['issues'].append(f"shape {img_a.shape} vs {img_b.shape}")
        return report  # voxelwise comparison impossible
    if not np.allclose(img_a.affine, img_b.affine, atol=1e-5):
        report['issues'].append("affine differs")
    if img_a.get_data_dtype() != img_b.get_data_dtype():
        report['issues'].append(
            f"dtype {img_a.get_data_dtype()} vs {img_b.get_data_dtype()}")

    a = np.asanyarray(img_a.dataobj)
    b = np.asanyarray(img_b.dataobj)
    diff_mask = a != b
    n_diff = int(diff_mask.sum())
    report['n_voxels'] = a.size
    report['n_diff'] = n_diff
    if n_diff == 0:
        return report

    report['pct_diff'] = 100.0 * n_diff / a.size
    is_labelmap = (np.issubdtype(a.dtype, np.integer)
                   and np.issubdtype(b.dtype, np.integer))
    if not is_labelmap:
        d = (a.astype(np.float64) - b.astype(np.float64))[diff_mask]
        report['value_diff'] = (float(np.abs(d).max()), float(np.abs(d).mean()))
        return report

    labels = np.union1d(np.unique(a), np.unique(b))
    report['fg_dice'] = dice(a > 0, b > 0)
    if dice_per_label:
        report['label_dice'] = {
            int(l): dice(a == l, b == l) for l in labels if l != 0}
    # top disagreeing label pairs (a_label -> b_label)
    pairs, counts = np.unique(
        np.stack([a[diff_mask], b[diff_mask]]), axis=1, return_counts=True)
    order = np.argsort(counts)[::-1][:10]
    report['top_confusions'] = [
        (int(pairs[0, i]), int(pairs[1, i]), int(counts[i])) for i in order]
    return report


def dice(mask_a, mask_b):
    denom = int(mask_a.sum()) + int(mask_b.sum())
    if denom == 0:
        return float('nan')
    return 2.0 * int((mask_a & mask_b).sum()) / denom


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('folder_a', type=Path)
    ap.add_argument('folder_b', type=Path)
    ap.add_argument('--dice-per-label', action='store_true',
                    help='print Dice for every nonzero label, not just those that disagree')
    args = ap.parse_args()

    files_a, files_b = find_niftis(args.folder_a), find_niftis(args.folder_b)
    only_a = sorted(set(files_a) - set(files_b))
    only_b = sorted(set(files_b) - set(files_a))
    common = sorted(set(files_a) & set(files_b))
    if only_a:
        print(f"only in {args.folder_a}: {only_a}")
    if only_b:
        print(f"only in {args.folder_b}: {only_b}")
    if not common:
        sys.exit("no common NIfTI files")

    n_identical = 0
    for name in common:
        r = compare_pair(files_a[name], files_b[name], args.dice_per_label)
        if r['issues']:
            print(f"{name}: HEADER MISMATCH — {'; '.join(r['issues'])}")
            if 'n_diff' not in r:
                continue
        if r['n_diff'] == 0:
            n_identical += 1
            if not r['issues']:
                print(f"{name}: identical ({r['n_voxels']:,} voxels)")
            continue
        print(f"{name}: {r['n_diff']:,}/{r['n_voxels']:,} voxels differ "
              f"({r['pct_diff']:.4f}%)")
        if 'fg_dice' in r:
            print(f"  foreground dice: {r['fg_dice']:.6f}")
        if 'value_diff' in r:
            mx, mean = r['value_diff']
            print(f"  |diff| over differing voxels: max {mx:g}, mean {mean:g}")
        if 'label_dice' in r:
            for l, d in r['label_dice'].items():
                print(f"  label {l:3d} dice: {d:.6f}")
        if 'top_confusions' in r:
            print("  top disagreements (label_a -> label_b: n):")
            for la, lb, n in r['top_confusions']:
                print(f"    {la:3d} -> {lb:3d}: {n:,}")

    print(f"\n{n_identical}/{len(common)} common files are voxel-identical")


if __name__ == '__main__':
    main()
