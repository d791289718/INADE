# python train.py --name INADE_GID_v3 --dataset_mode GID --norm_mode inade  --dataroot datasets --batchSize 8  --niter 100 --niter_decay 100 --tf_log --train_eval --split_dir './datasets/GID/split_txt_v3' --use_vae
# --use_segmodel 记得也要把batchsize调整为10

# python train_deeplabv3.py --name v3_baseline --dataset_mode GID --batchSize 42 --niter 200 --checkpoints_dir "deep_lab/checkpoints/" --results_dir "deep_lab/results/"
# python train_deeplabv3.py --name v3_aug_v9_trainrealval_2 --dataset_mode GID --batchSize 42 --niter 200 --checkpoints_dir "deep_lab/checkpoints/" --results_dir "deep_lab/results/" --add_aug --aug_txt 'v9_trainrealval_2.txt'

# python train_unet.py --name unet_v3_baseline_aug --dataset_mode GID --batchSize 42 --niter 200 --checkpoints_dir "deep_lab/checkpoints/" --results_dir "deep_lab/results"
# python train_unet.py --name unet_v3_segfrozen_aug_new_v5_train_1 --dataset_mode GID --batchSize 42 --niter 200 --checkpoints_dir "deep_lab/checkpoints/" --results_dir "deep_lab/results/" --add_aug --aug_txt 'new_v5_train_1.txt'


# ============================================= cityscapes =============================================
python train.py --name city_baseline --dataset_mode cityscapes --norm_mode inade  --dataroot datasets/cityscapes --batchSize 6  --niter 100 --niter_decay 100 --tf_log --train_eval --use_segmodel