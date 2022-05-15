# python train.py --name GID_v3_VAE --dataset_mode GID --norm_mode inade  --dataroot datasets --batchSize 10  --niter 100 --niter_decay 100 --tf_log --train_eval --use_vae
# python train_deeplabv3.py --name v3_baseline --dataset_mode GID --batchSize 42 --niter 200 --checkpoints_dir "deep_lab/checkpoints/" --results_dir "deep_lab/results/"
python train_deeplabv3.py --name v3_woaug_v4_realwoaug_1.txt --dataset_mode GID --batchSize 42 --niter 200 --checkpoints_dir "deep_lab/checkpoints/" --results_dir "deep_lab/results/" --add_aug --aug_txt 'v4_realwoaug_1.txt'
