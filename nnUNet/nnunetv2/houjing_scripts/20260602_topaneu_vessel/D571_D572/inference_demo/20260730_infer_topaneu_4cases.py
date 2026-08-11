"""Usage:
# Shifted to new system where uv manages packages.
# No conda anymore.

cd /mnt/x/data2/Project/TopCoW_Algo_Submission/task-1-seg/nnUNet_TopCoW
source .venv/bin/activate

systemd-run --user --scope -p CPUWeight=50 -p MemoryHigh=96G -p MemoryMax=128G -p IOWeight=30 \
  env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  nnUNet_n_proc_DA=1 CUDA_VISIBLE_DEVICES=0 \
  python nnunetv2/houjing_scripts/20260602_topaneu_vessel/D571_D572/20260706_topaneu_batch2/20260730_infer_topaneu_4cases.py

We don't use dataloader of nnunetv2, so set nnUNet_n_proc_DA=1.
"""
from nnunetv2.houjing_scripts.infer_ppl_parallel_npz import infer_folder
from glob import glob
import os

def infer_one_model(model_cfg, in_dir, out_dir, suffix='.nii.gz', output_ext='.nii.gz', alias=None, sequential=False, n_infer_workers=1, n_pre_post_workers=2, n_gpus=1, overwrite_trainer_name=None):
    if alias is None:
        alias = os.path.basename(out_dir)
    print(f"\n\n=== Inferring {alias} ===\n\n")
    files = sorted(glob(f"{in_dir}/*{suffix}"))
    print(f"Found {len(files)} files")
    
    fnames = []
    fnames += [os.path.basename(x).replace(suffix, '') for x in files]
    print(f"\n {len(fnames) = } \n")
    
    # If the checkpoint contains unrecognized trainer_name, you can set an existing trainer_name 
    #   if it can init the same model arch as the trained one.
    if overwrite_trainer_name:
        os.environ['trainer_name'] = overwrite_trainer_name
    infer_folder(
        in_dir=in_dir,
        out_dir=out_dir,
        fnames=fnames,
        suffix=suffix,
        output_ext=output_ext,
        model_cfg=model_cfg,
        sequential=sequential,
        n_preprocess_workers=n_pre_post_workers,
        n_infer_workers=n_infer_workers,
        n_post_inference_workers=n_pre_post_workers,
        queue1_size=n_pre_post_workers,
        queue2_size=n_pre_post_workers,
        n_gpus=n_gpus,
        use_mirroring=True,
        post_process=True,
        skip_existing=True
    )
    if overwrite_trainer_name:
        del os.environ['trainer_name']

def infer():
    out_base_dir = f"data/results/20260730_infer_topaneu"

    # nnUNet v2.8
    # Trained on OMEN
    # NOTE: these OMEN models were patched in-place by 20260611_patch_omen_v2.8_models_for_v2.5_infer.py
    # (checkpoint trainer_name rewritten to a repo trainer; _primus added to Primus plans.json), so they
    # need NO `overwrite_trainer_name` and NO PRIMUS_* env vars -> safe to mix in an ensemble.

    in_dir='data/raw/20260730_4cases/new4casesImages'

    # big ensemble

    model_cfg = []
    model_cfg += [
        dict(
            base_model_dir=f'data/results/Dataset572_TopAneu_Vessel_36fgCls_wLRSwap',
            subdir='resEncM_ceW_bg0.75_max1.5_diff_cluster_noMirror_bs2_ps80_192_128_allLR1e-2_clsBalSamp_degree0.75_noTopcowPretrain_ep1000_DS3',
            model_fold=4, ckpt_name='checkpoint_final.pth', patch_size=None, tile_step_size=0.5
        )
    ]
    model_cfg += [
        dict(
            base_model_dir=f'data/results/Dataset572_TopAneu_Vessel_36fgCls_wLRSwap',
            subdir='plain_conv_ceW_bg0.5_max2.5_diff_cluster_noMirror_bs2_ps80_192_128_allLR1e-2_clsBalSamp_degree0.75_noTopcowPretrain_ep1000_DS3',
            model_fold=4, ckpt_name='checkpoint_final.pth', patch_size=None, tile_step_size=0.5
        )
    ]
    model_cfg += [
        dict(
            base_model_dir=f'data/results/Dataset572_TopAneu_Vessel_36fgCls_wLRSwap',
            subdir='primusV3S_ceW_bg0.5_max2.5_diff_cluster_noMirror_bs2_ps80_192_128_clsBalSamp_degree0.75_warm50_ep1000_DS',
            model_fold=4, ckpt_name='checkpoint_final.pth', patch_size=None, tile_step_size=0.5
        )
    ]
    infer_one_model(
        model_cfg=model_cfg,
        in_dir=in_dir,
        out_dir=os.path.join(out_base_dir, 'ensemble_v2.5___resDS3___convDS3___primusDS2', '20260730_topaneu_vessel_36cls_pred'),
    )


if __name__ == "__main__":
    infer()