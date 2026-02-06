#!/bin/bash
export WANDB_DISABLED=true
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
n_splits=3
setting=semi_supervised
augmentation='yes'
TRAIN_GPUS="2,3"
INFERENCE_GPUS="2,3"
n_train_node=2
n_test_node=2
n_permutations=21

for model in 'smol'; do
    batch_size=4
    eval_batch_size=$((batch_size*2))
    for ((idx = 0 ; idx < 6 ; idx++ )); do  
        #dataset=20news-$idx
        dataset=20news-3
        expdir=exp/$dataset/$setting/split$n_splits
        for ((split_idx = 0; split_idx < $n_splits ; split_idx++ )); do    
            CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting --max_steps 2000\
                                                        --batch_size $batch_size --model $model --binning standard  --lr 0.0005 --augmentation $augmentation
            CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS torchrun --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting\
                                                    --batch_size $eval_batch_size  --n_permutations $n_permutations --model $model --binning standard  --lr 0.0005 --augmentation $augmentation
            python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting --split_idx $split_idx --augmentation $augmentation| tee $expdir/evaluate.log
        done
        python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting  --augmentation $augmentation| tee $expdir/evaluate.log
    done

done


