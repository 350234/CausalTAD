#!/bin/bash
export WANDB_DISABLED=true
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
n_splits=5
setting=semi_supervised
augmentation='yes'
TRAIN_GPUS="0"
INFERENCE_GPUS="0"
n_train_node=1
n_test_node=1
n_permutations=21

for model in 'smol'; do
    batch_size=4
    eval_batch_size=$((batch_size*2))
    for dataset in 'lymphography'; do
        expdir=exp/$dataset/$setting/split$n_splits
        CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx 0 --setting $setting --max_steps 2000 \
                                                    --batch_size $batch_size --model $model --binning standard --augmentation $augmentation
        CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS torchrun --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx 0  --setting $setting\
                                                --batch_size $eval_batch_size  --n_permutations $n_permutations --model $model --binning standard --augmentation $augmentation
        python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting --split_idx 0 --augmentation $augmentation| tee $expdir/evaluate.log  
        for ((split_idx = 1 ; split_idx < $n_splits ; split_idx++ )); do    
            CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting --max_steps 2000\
                                                        --batch_size $batch_size --model $model --binning standard  --augmentation $augmentation
            CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS  torchrun --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting\
                                                    --batch_size $eval_batch_size  --n_permutations $n_permutations --model $model --binning standard --augmentation $augmentation
            python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting --split_idx $split_idx --augmentation $augmentation| tee $expdir/evaluate.log   
        done
        python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting | tee $expdir/evaluate.log

    done

    # batch_size=4
    # eval_batch_size=$((batch_size*2))
    # for dataset in 'fakejob'; do
    #     expdir=exp/$dataset/$setting/split$n_splits
    #     CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx 0 --setting $setting --max_steps 20000 \
    #                                                 --batch_size $batch_size --model $model --binning standard
    #     CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS  torchrun --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx 0  --setting $setting\
    #                                             --batch_size $eval_batch_size  --n_permutations $n_permutations --model $model --binning standard   
    #     for ((split_idx = 1 ; split_idx < $n_splits ; split_idx++ )); do    
    #         CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting --max_steps 20000\
    #                                                     --batch_size $batch_size --model $model --binning standard  
    #         CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS  torchrun --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting\
    #                                                 --batch_size $eval_batch_size  --n_permutations $n_permutations --model $model --binning standard   
    #     done
    #     python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting | tee $expdir/evaluate.log
    # done

done


