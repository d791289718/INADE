for ((i=100; i>=5; i-=5))
do
    echo $i
    python test.py --name GID_INADE --norm_mode inade --batchSize 20 --gpu_ids 0 --which_epoch best --dataset_mode GID --phase "val"
done