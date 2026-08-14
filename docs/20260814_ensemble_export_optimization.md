# Ensemble Export Optimization: Fusion Order, Slab Streaming, fp16

*2026-08-14. Applies to `run_inference.py` + `nnunetv2/houjing_scripts/infer_ppl_parallel_npz.py`
(3-model ensemble: resEncM + plain_conv + primusV3S, Dataset572, 36 fg classes, fold 4).*

## Problem

Per case, the original export stage ran `convert_predicted_logits_to_prob_with_correct_shape`
once **per model**: resample logits to full resolution → softmax → revert crop, then
averaged the three full-resolution 37-channel float32 probability maps and argmaxed.
For a typical MR case (198×572×470) each full-res probability map is ~7.9 GB; the
worst instant held the running sum plus the convert internals of the current model
(~24–30 GB). This forced `--memory=64g` for the parallel docker pipeline (32 GB
swapped/stalled on large CTs) and made export ~3× slower than needed.

## Fact 1: resample and ensemble-mean commute

`mean_i(R(p_i)) = R(mean_i(p_i))` holds exactly when R is linear and applied with
identical geometry to every model. Both hold here:

- R is trilinear interpolation (`resample_torch_fornnunet`, `mode='linear'`,
  effectively `F.interpolate(..., mode='trilinear', align_corners=False, antialias=False)`);
  the separate-z branch (linear in-plane + nearest across z) is also linear.
- All three models' plans share target spacing `[0.6000016, 0.375, 0.375]`,
  transpose `[0,1,2]`, ZScore normalization; crop bbox comes from the same image
  through the same code. Verified empirically: identical preprocessed shapes per case.

Caveat: the pipeline resamples **logits** and softmaxes after. Softmax is nonlinear,
so it does not commute with R — moving the fusion before the resample changes the
output slightly (see measurements).

## Fusion-order measurements (3 cases: 2 MR + 1 CT)

Per case, per model, logits were predicted once; three fusions were compared:

- **A (original)**: per model `softmax(R(logits))`, then arithmetic mean → argmax.
- **B (`--fuse_logits`)**: `softmax(R(mean logits))` — one resample. Softmax-of-mean-logits
  ≈ geometric-mean fusion: more conservative where models disagree.
- **C**: `R(mean softmax(logits))` — softmax before resample, arithmetic mean kept.

| comparison | differing voxels | character |
|---|---|---|
| A vs B | 2.4k–3.4k = 1.6–1.8 % of fg-union, 0.003–0.006 % of volume | ~99 % on structure boundaries (≤1 voxel shifts); median top1–top2 margin ≈ 0.10–0.14 (low-confidence voxels); transitions dominated by fg→background |
| A vs C | 4.5–5.3 % of fg-union | the softmax↔resample order swap matters *more* than the fusion-semantics change — C is both slower than B and further from A, so B is the right fast variant |

After production post-processing, per-class Dice(A,B): mean 0.93–0.98. Raw Dice-0
classes were 7–13-voxel remnants (Pcom/AICA/AChA) that post-processing removes from
A anyway. One substantive disagreement observed: 3rd-A3 vs L-A3 relabeling on an
ambiguous case (999→271 voxels). If recall on tiny communicating arteries is
critical, prefer the exact default over `--fuse_logits`.

Export time (3 resamples → 1): 32→10 s (MR), 51→17 s (CT), ~3.1×.

## Fact 2: the export tail is streamable in z-slabs (exact)

Everything after the sliding-window prediction is either **local along z**
(trilinear interpolation: each output slice needs ≤2 source slices) or
**per-voxel across channels** (softmax, ensemble accumulation, argmax, canvas
write). Therefore the whole tail

```
resample → softmax → accumulate over models → argmax → write into canvas
```

can run slab-by-slab (default slab budget 256 MB), models-outer, and no
full-resolution 37-channel float volume needs to exist per model. Only the shared
accumulator survives — and with `--fuse_logits` (single source), not even that:
slabs are argmaxed directly (softmax skipped: monotone per voxel, argmax unchanged;
same for the `/n` division).

Implementation (`_resample_slab_trilinear`, `_streamed_seg_from_sources`):

- Trilinear is separable: 1D z-lerp (manual `align_corners=False` coordinate map
  `src = (dst + 0.5)·in/out − 0.5`, clamped) + 2D bilinear (`F.interpolate` on the
  slab, all channels in one reshaped call). Matches full-volume `F.interpolate`
  to ≤6e-6 (float rounding order), zero argmax mismatches in unit tests.
- Crop revert without a float canvas: slabs are written into a background(=0)-filled
  uint8 canvas at the bbox offset — equivalent to `revert_cropping_on_probabilities`
  (background prob 1 outside bbox) followed by argmax.
- Falls back automatically to the full-volume path when the plans would use
  separate-z resampling or region-based labels (`streamed=False` also forces it).

### Per-channel vs per-slab chunking

The codebase already chunked the resample **per channel** ("HOUJING: per-channel
interpolation" in `resample_torch_simple`). Same generic idea, different axis, and
the axis determines what can be fused downstream:

- **Per channel**: channels are independent in interpolation (trivially exact,
  no coordinate math), but softmax/argmax need *all* channels of a voxel, so the
  output must still be the full 37-channel full-res volume. Only the transient
  peak inside the resample call shrinks.
- **Per z-slab**: each chunk is complete across channels, so the entire tail
  fuses into the slab loop and each slab is discarded immediately. Costs the
  manual coordinate mapping and exactness only up to rounding order (~1e-6).

They compose (channels within a slab), but a ≤256 MB slab makes that unnecessary —
one bilinear call per slab is also faster than 37 small ones. The old per-channel
loop now only matters in the fallback paths.

Could per-channel match per-slab's peak? Yes — with progressive reclamation, a
per-channel scheme reaches the same complexity class (one 37-channel volume +
accumulator at peak): resample channel by channel while freeing each consumed
source channel (source shrinks as the full-res buffer grows, so the phase peaks
at exactly one complete full-res buffer), softmax that buffer in place, add it
into the accumulator in place. Single resample pass, stock per-channel
`F.interpolate` (bit-exact per channel), no online-softmax machinery needed.
Two caveats versus per-slab:

- **Constant factor**: the volume coexisting with the accumulator is the
  *full-res* one rather than the *preprocessed* one. With these models
  (preprocessing coarser than the originals, so full-res has more voxels):
  MR 7.9 vs 4.5 GB, CT 11.7 vs 9.1 GB → peak ≈ 15.7 / 23.8 GB vs the measured
  13.4 / 21.9 GB for per-slab. Same class, worse constants here; the comparison
  narrows or flips where preprocessing upsamples less.
- **Tensor granularity**: freeing "one channel at a time" is impossible on the
  contiguous logits tensor the sliding-window predictor returns — a slice of one
  allocation cannot be released. True per-channel reclamation requires the
  predictor to accumulate into 37 separate per-channel tensors (surgery inside
  `predict_sliding_window_return_logits`); otherwise source + growing buffer +
  accumulator briefly coexist (~16.3 GB MR, worse than per-slab).

Per-slab was chosen because softmax/mean/argmax fuse into channel-complete slabs
with zero extra state, in a single pass, without touching the predictor.

## fp16 (default ON, `--no-fp16` to disable)

`--fp16` keeps the ensemble probability accumulator in float16 (exact path), or the
fused logits in float16 (`--fuse_logits` path). Probabilities in [0,1] summed over
3 models are comfortably within fp16; argmax flips only at near-exact ties.
Measured: 20–45 differing voxels per case (~5e-7 of the volume) — at or below the
run-to-run GPU nondeterminism baseline (13–31 voxels between two runs of identical
code).

## Pitfall: generator frames keep the previous logits alive

The models-outer stream consumes logits from a generator (`iter_logits`), so only
one model's logits should be resident at a time. But a generator frame keeps its
locals alive across `yield`: the frame's `logits` name was only *reassigned* when
the **next** model's prediction returned, so model k's tensor (4.5 GB MR / 9.1 GB
CT) silently survived throughout model k+1's entire sliding-window prediction —
two logits coexisting exactly during the memory-heaviest phase. The consumer's
`del src` cannot help; the fix is `del logits` in the generator immediately after
the `yield` (it executes on resume, before the next prediction starts). Worth
remembering for any "one item resident at a time" generator holding large arrays.

## Peak RSS / time (single process, whole pipeline incl. model load + preprocess + inference)

MR case 198×572×470, CT case 301×512×512. Modes above the line were measured
before the generator fix; the default was re-measured after it:

| mode | MR RSS | CT RSS | MR predict+export | CT predict+export |
|---|---|---|---|---|
| full-volume (old) | 29.6 GB | 44.7 GB | 48 s | 80 s |
| streamed (pre-fix) | 21.3 GB | 35.8 GB | 40 s | 87 s |
| streamed + fp16 (pre-fix) | 17.6 GB | 30.3 GB | 42 s | 89 s |
| **streamed + fp16 + generator fix (default)** | **13.4 GB** | **21.9 GB** | 40 s | 85 s |
| `--fuse_logits` (pre-fix) | 18.0 GB | 33.1 GB | 25 s | 49 s |
| `--fuse_logits --fp16` (pre-fix) | 16.1 GB | 29.2 GB | 23 s | 57 s |

The generator-fix reductions (−4.2 GB MR, −8.4 GB CT) match one logits tensor
exactly. Remaining floor: sliding-window internals (CPU-side logits 4.5 GB MR /
9.1 GB CT) plus ~2.5 GB weights — export is no longer the memory bottleneck.

### Peak anatomy of the default (exact streamed) path

For models 2/3 of the outer loop, two big tensors necessarily coexist — this is
the peak, and it is irreducible within the models-outer design:

| resident tensor | grid | dtype | MR | CT |
|---|---|---|---|---|
| accumulator | full-res cropped (`shape_after_cropping_and_before_resampling`) | fp16 | 3.9 GB | 5.8 GB |
| current model's logits (sliding-window output) | preprocessed | fp32 | 4.5 GB | 9.1 GB |
| slab transients | slab | fp32 | ~0.5 GB | ~0.5 GB |
| baseline (weights, preprocessed copies, interpreter) | — | — | ~4.9 GB | ~6.3 GB |
| **sum vs measured peak** | | | ≈13.3 / **13.4** | ≈21.2 / **21.9** |

Model 1's phase is cheaper (accumulator not yet allocated). `--fuse_logits`
replaces the accumulator with an fp16 logits sum on the preprocessed grid — no
full-res float at all, hence the lowest-memory mode. If the peak ever needs to go
lower, the next lever is keeping each model's logits on the GPU (VRAM peak ~7–10
GB vs the 16 GB cap) and gathering resample slabs directly from GPU memory,
leaving only the accumulator + slabs in RAM.

Note: the *container-level* peak of the parallel docker run barely moved with the
generator fix (19.3 → 19.1 GiB on the 5 debug samples). The parallel pipeline's
peak is set by concurrent phases across its worker processes (inference worker on
one case while the preprocess/post workers hold neighbors), not by the
intra-process overlap the fix removed — the fix mainly lowers the single-process /
sequential peak and the inference worker's own footprint on the largest CTs.

Output equivalence (against the 13–31-voxel rerun-noise baseline): streamed vs
full-volume 16/21 voxels; parallel end-to-end vs pre-change reference 14 voxels —
i.e. equivalent.

## Docker validation (new default)

5 debug samples (`20260812_DEBUG_MEM/selected_5images`, 3 MR + 2 CT incl. the
largest CTs), `--memory=32g --shm-size=32g`, `--gpu_limit_GB 16`, GPU 1:

- exit 0, all 5 segmentations written; wall 502 s (~100 s/case, host concurrently loaded)
- container RAM: peak **19.3 GiB**, median 13.5 GiB (old default OOM/stalled at 32 GB)
- VRAM: peak **6.8 GiB** under the 16 GB cap

README updated accordingly: 32 GB is the validated default; the sequential mode is
now only a low-VRAM (<16 GB GPU) fallback.

## Flags summary

| flag | default | effect |
|---|---|---|
| `--fp16` / `--no-fp16` | on | fp16 accumulator / fused logits; diffs below GPU noise |
| `--fuse_logits` | off | softmax-of-mean-logits fusion, 1 resample; ~2× faster, slightly conservative at boundaries; drops tiny uncertain structures slightly more often. Only valid when all models share the same preprocessing target spacing/transpose (true for the built-in ensemble; enforced at runtime — matching shapes alone are no proof, different spacings can round to the same shape) |
| (internal) `streamed` | on | slab-streamed export; auto-fallback for separate-z plans / region labels |
