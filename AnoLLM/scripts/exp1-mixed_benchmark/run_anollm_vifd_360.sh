#!/bin/bash
PORT_LOCK_FILE=/var/lock/anollm_port.lock

reserve_free_port() {
  # Use file descriptor 9 with flock for mutual exclusion
  exec 9> "$PORT_LOCK_FILE"
  flock -x 9

  for p in $(seq 29500 29999); do
    # Check whether the port is already listening
    if ! ss -ltn | awk '{print $4}' | grep -Eq "(:|\\[)::?$p$|:$p$"; then
      # Reserve immediately (write to temp file for debugging/tracking)
      echo "$p" > /tmp/last_reserved_anollm_port
      # release lock and return
      flock -u 9
      exec 9>&-
      echo "$p"
      return 0
    fi
  done

  flock -u 9
  exec 9>&-
  return 1
}
export WANDB_DISABLED=true
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
n_splits=5
setting=semi_supervised
augmentation='yes_a'
TRAIN_GPUS="1,0"
INFERENCE_GPUS="1,0"
n_train_node=2
n_test_node=2
n_permutations=21
abnormal_ratio=0.25
max_steps=2000
models=('smol-360')
lr=5e-5
graph_based_rank='no'
score_weight='yes'
sorted_set_path='vifd.json'

for model in "${models[@]}"; do
    batch_size=16
    eval_batch_size=$((batch_size*8))
    for dataset in 'vifd'; do
        expdir=exp/$dataset/$setting/split$n_splits
        PORT=$(reserve_free_port) || { echo "no free port"; exit 1; }
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=$PORT
        echo "Reserved MASTER_PORT=$PORT"
        CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:$PORT --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx 0 --setting $setting --max_steps $max_steps \
                                                    --batch_size $batch_size --model $model --binning standard  --lr $lr --augmentation $augmentation --abnormal_ratio $abnormal_ratio
        PORT=$(reserve_free_port) || { echo "no free port"; exit 1; }
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=$PORT
        echo "Reserved MASTER_PORT=$PORT"
        CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:$PORT --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx 0  --setting $setting\
                                                --batch_size $eval_batch_size  --n_permutations $n_permutations --lr $lr --model $model --binning standard --augmentation $augmentation --abnormal_ratio $abnormal_ratio
        python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting --split_idx 0 --augmentation $augmentation| tee $expdir/evaluate.log
        for ((split_idx = 1 ; split_idx < $n_splits ; split_idx++ )); do
            PORT=$(reserve_free_port) || { echo "no free port"; exit 1; }
            export MASTER_ADDR=127.0.0.1
            export MASTER_PORT=$PORT
            echo "Reserved MASTER_PORT=$PORT"
            CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:$PORT --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting --max_steps $max_steps\
                                                        --batch_size $batch_size --model $model --binning standard  --lr $lr --augmentation $augmentation --abnormal_ratio $abnormal_ratio
            PORT=$(reserve_free_port) || { echo "no free port"; exit 1; }
            export MASTER_ADDR=127.0.0.1
            export MASTER_PORT=$PORT
            echo "Reserved MASTER_PORT=$PORT"
            CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS  torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:$PORT --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting\
                                                    --batch_size $eval_batch_size  --n_permutations $n_permutations --lr $lr --model $model --binning standard --augmentation $augmentation --abnormal_ratio $abnormal_ratio
            python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting --split_idx $split_idx --augmentation $augmentation| tee $expdir/evaluate.log   
        done
        python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting --augmentation $augmentation| tee $expdir/evaluate.log

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


