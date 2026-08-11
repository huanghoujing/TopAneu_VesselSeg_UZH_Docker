# xvfb-run -a ./.venv/bin/python \
#     nnunetv2/houjing_scripts/vis_label_screenshots_napari_multi_view.py \
#     --labels_dir data/results/20260708_infer_topaneu/ensemble_v2.5___resDS3___convDS3___primusDS2/20260708_topaneu_vessel_36cls_pred \
#     --images_dir data/raw/20260708_batch_1_2_updated/Train/images \
#     --out_dir    data/results/20260708_infer_topaneu_VIZ_3View_3Mid \
#     --num_samples -1 --shuffle \
#     --grid_cols 3 \
#     --views anterior left superior x y z \
#     --skip_existing

# xvfb-run -a ./.venv/bin/python \
#     nnunetv2/houjing_scripts/vis_label_screenshots_napari_multi_view.py \
#     --labels_dir data/results/20260728_infer_topaneu/ensemble_v2.5___resDS3___convDS3___primusDS2/20260728_topaneu_vessel_36cls_pred \
#     --images_dir data/raw/20260728_mr_436/images \
#     --out_dir    data/results/20260728_infer_topaneu_VIZ_3View_3Mid \
#     --num_samples -1 --shuffle \
#     --grid_cols 3 \
#     --views anterior left superior x y z \
#     --skip_existing

xvfb-run -a ./.venv/bin/python \
    nnunetv2/houjing_scripts/vis_label_screenshots_napari_multi_view.py \
    --labels_dir data/results/20260730_infer_topaneu/ensemble_v2.5___resDS3___convDS3___primusDS2/20260730_topaneu_vessel_36cls_pred \
    --images_dir data/raw/20260730_4cases/new4casesImages \
    --out_dir    data/results/20260730_infer_topaneu_VIZ_3View_3Mid \
    --num_samples -1 --shuffle \
    --grid_cols 3 \
    --views anterior left superior x y z \
    --skip_existing