set -eux

echo ${nnUNet_results}

export PRINT_NETWORK=1
export segmentation_export_pool_size=5
export num_ds_levels=3

# DATASET_NAME=Dataset571_TopAneu_Vessel_36fgCls
DATASET_NAME=Dataset572_TopAneu_Vessel_36fgCls_wLRSwap

# plainConv 1000 epochs
nnUNet_n_proc_DA=8 \
    CUDA_VISIBLE_DEVICES=0 \
    n_classes_w_bg=37 \
    ce_class_weight=0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,2.5,2.5,1.0,1.0,1.0,1.5,2.5,2.5,1.0,1.0,1.0,1.0,1.5,1.0,1.0,1.0,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,1.0,1.0 \
    weight_diff=1 weight_lcluster=1 \
    MIRROR_AXES=none \
    batch_dice=0 dice_do_bg=0 \
    OVERSAMPLE_FOREGROUND_PERCENT=0.75 cls_balanced_global_sampling=1 fg_class_calibration_degree=0.75 \
    enable_deep_supervision=1  \
    num_epochs_per_val=100 \
    nnUNetv2_train ${DATASET_NAME} 3d_fullres 4 \
    -tr Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg \
    --num_epochs 1000 \
    --initial_lr 1e-2 --cls_lr 1e-2 \
    --output_folder_base ${nnUNet_results}/${DATASET_NAME}/plain_conv_ceW_bg0.5_max2.5_diff_cluster_noMirror_bs2_ps80_192_128_allLR1e-2_clsBalSamp_degree0.75_noTopcowPretrain_ep1000_DS${num_ds_levels} --c
