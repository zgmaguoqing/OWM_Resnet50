#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task 2
#SBATCH --nodelist=n18
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH -o slurm%A_%a_out.log
#SBATCH -e slurm%A_%a_err.log
#SBATCH --array=0-13%1

cmdlines=(
    "python iBatchLearn.py --cfg ./4splitDomains.yaml --user_cfg ./utils/user_4splitDomains.yaml --task_count 0 \
        --suffix local_test --init_path ./init_model/4splitDomains.pth --ckpt_path ./init_model/4splitDomains.pth --save_ckpt_path ./model_info/4splitDomains/checkpoint-0.pth \
        --save_storage_path ./model_info/4splitDomains/storage-0.pth"
    "python iBatchLearn.py --cfg ./4splitDomains.yaml --user_cfg ./utils/user_4splitDomains.yaml --task_count 1 \
        --suffix local_test --init_path ./init_model/4splitDomains.pth --ckpt_path ./model_info/4splitDomains/checkpoint-0.pth --save_ckpt_path ./model_info/4splitDomains/checkpoint-1.pth \
        --storage_path ./model_info/4splitDomains/storage-0.pth --save_storage_path ./model_info/4splitDomains/storage-1.pth"
    "python iBatchLearn.py --cfg ./4splitDomains.yaml --user_cfg ./utils/user_4splitDomains.yaml --task_count 2 \
        --suffix local_test --init_path ./init_model/4splitDomains.pth --ckpt_path ./model_info/4splitDomains/checkpoint-1.pth --save_ckpt_path ./model_info/4splitDomains/checkpoint-2.pth \
        --storage_path ./model_info/4splitDomains/storage-1.pth --save_storage_path ./model_info/4splitDomains/storage-2.pth"
    "python iBatchLearn.py --cfg ./4splitDomains.yaml --user_cfg ./utils/user_4splitDomains.yaml --task_count 3 \
        --suffix local_test --init_path ./init_model/4splitDomains.pth --ckpt_path ./model_info/4splitDomains/checkpoint-2.pth --save_ckpt_path ./model_info/4splitDomains/checkpoint-3.pth \
        --storage_path ./model_info/4splitDomains/storage-2.pth --save_storage_path ./model_info/4splitDomains/storage-3.pth"
    
    "python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 0 \
        --suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./init_model/10splitTasks.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-0.pth \
        --save_storage_path ./model_info/10splitTasks/storage-0.pth"
    "python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 1 \
        --suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./model_info/10splitTasks/checkpoint-0.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-1.pth \
        --storage_path ./model_info/10splitTasks/storage-0.pth --save_storage_path ./model_info/10splitTasks/storage-1.pth"
    "python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 2 \
        --suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./model_info/10splitTasks/checkpoint-1.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-2.pth \
        --storage_path ./model_info/10splitTasks/storage-1.pth --save_storage_path ./model_info/10splitTasks/storage-2.pth"
    "python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 3 \
        --suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./model_info/10splitTasks/checkpoint-2.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-3.pth \
        --storage_path ./model_info/10splitTasks/storage-2.pth --save_storage_path ./model_info/10splitTasks/storage-3.pth"
    "python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 4 \
        --suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./model_info/10splitTasks/checkpoint-3.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-4.pth \
        --storage_path ./model_info/10splitTasks/storage-3.pth --save_storage_path ./model_info/10splitTasks/storage-4.pth"
    
    "python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 5 \
        --suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./model_info/10splitTasks/checkpoint-4.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-5.pth \
        --storage_path ./model_info/10splitTasks/storage-4.pth --save_storage_path ./model_info/10splitTasks/storage-5.pth"
    "python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 6 \
        --suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./model_info/10splitTasks/checkpoint-5.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-6.pth \
        --storage_path ./model_info/10splitTasks/storage-5.pth --save_storage_path ./model_info/10splitTasks/storage-6.pth"
    "python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 7 \
        --suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./model_info/10splitTasks/checkpoint-6.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-7.pth \
        --storage_path ./model_info/10splitTasks/storage-6.pth --save_storage_path ./model_info/10splitTasks/storage-7.pth"
    "python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 8 \
        --suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./model_info/10splitTasks/checkpoint-7.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-8.pth \
        --storage_path ./model_info/10splitTasks/storage-7.pth --save_storage_path ./model_info/10splitTasks/storage-8.pth"
    "python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 9 \
        --suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./model_info/10splitTasks/checkpoint-8.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-9.pth \
        --storage_path ./model_info/10splitTasks/storage-8.pth --save_storage_path ./model_info/10splitTasks/storage-9.pth"
    
)
cmd=${cmdlines[SLURM_ARRAY_TASK_ID]}
$cmd

