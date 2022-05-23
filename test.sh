# for ((i=145; i>=90; i-=10))
# do
#     echo $i
#     python test.py --name GID_baseline --norm_mode inade --batchSize 20 --gpu_ids 0 --which_epoch $i --dataset_mode GID --phase "test"
# done
# python test.py --name GID_baseline --norm_mode inade --batchSize 20 --gpu_ids 0 --which_epoch best --dataset_mode GID --phase "test"

# python test_deeplabv3.py --name v3_aug_v9_trainrealval_2 --batchSize 200 --dataset_mode GID --phase val --checkpoints_dir "deep_lab/checkpoints/" --results_dir "deep_lab/results/"

python test_unet.py --name unet_v3_aug_v2_train_1 --batchSize 40 --dataset_mode GID --phase val --checkpoints_dir "deep_lab/checkpoints/" --results_dir "deep_lab/results/"