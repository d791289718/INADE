python train.py --name GID_v3_segworeal --dataset_mode GID --norm_mode inade  --dataroot datasets --batchSize 10  --niter 100 --niter_decay 100 --tf_log --train_eval --use_segmodel --split_dir './datasets/GID/split_txt_v3' --no_realimg_seg_loss

# python train_deeplabv3.py --name v3_baseline --dataset_mode GID --batchSize 42 --niter 200 --checkpoints_dir "deep_lab/checkpoints/" --results_dir "deep_lab/results/"
# python train_deeplabv3.py --name v3_aug_v9_trainrealval_2 --dataset_mode GID --batchSize 42 --niter 200 --checkpoints_dir "deep_lab/checkpoints/" --results_dir "deep_lab/results/" --add_aug --aug_txt 'v9_trainrealval_2.txt'

# python train_unet.py --name unet_v3_baseline_aug --dataset_mode GID --batchSize 42 --niter 200 --checkpoints_dir "deep_lab/checkpoints/" --results_dir "deep_lab/results"
# python train_unet.py --name unet_v3_aug_v2_train_1 --dataset_mode GID --batchSize 42 --niter 200 --checkpoints_dir "deep_lab/checkpoints/" --results_dir "deep_lab/results/" --add_aug --aug_txt 'v2_train_1.txt'