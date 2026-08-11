python nnunetv2/houjing_scripts/count_label_n_voxels_and_spacing.py \
    --input_dir /mnt/x/data2/nnunet_data/raw/Dataset571_TopAneu_Vessel_36fgCls/topaneu_topbrain_data_Jun022026FINAL/labelsTr \
    --max_label_value 36 \
    --output_csv_file /mnt/x/data2/nnunet_data/raw/Dataset571_TopAneu_Vessel_36fgCls/topaneu_vessel_label_n_voxels.csv \
    --num_workers 3
