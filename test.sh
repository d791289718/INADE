# for ((i=145; i>=90; i-=10))
# do
#     echo $i
#     python test.py --name GID_baseline --norm_mode inade --batchSize 20 --gpu_ids 0 --which_epoch $i --dataset_mode GID --phase "test"
# done
# python test.py --name GID_baseline --norm_mode inade --batchSize 20 --gpu_ids 0 --which_epoch best --dataset_mode GID --phase "test"

python test_deeplabv3.py --name v3_aug_v1_trainreal_2 --batchSize 200 --dataset_mode GID --phase val --checkpoints_dir "deep_lab/checkpoints/" --results_dir "deep_lab/results/"