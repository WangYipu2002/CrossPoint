export PYTHONPATH=$(pwd)/train

# Multi-GPU communication related environment variables (comment out for single GPU training)
export NCCL_SOCKET_IFNAME=eth0      # or bond0, check ifconfig output
export NCCL_IB_DISABLE=1            # Disable InfiniBand
export NCCL_DEBUG=INFO              # Can be enabled for debugging

export OMP_NUM_THREADS=4
export WANDB_MODE=offline
export ACCELERATE_CPU_AFFINITY=1

# You should edit the following paths
export MEDIA_DIR=/your/path/to/training_data/directory
export PRETRAIN_MODEL_PATH=/your/path/to/pretained_model
export OUTPUT_PATH=/your/path/to/output/checkpoints
export DATASET=CrossPoint-378K # change the dataset name in data/dataset_info.json 

if [ ! -d "$OUTPUT_PATH" ]; then
  mkdir "$OUTPUT_PATH"
fi

torchrun --nproc_per_node=8 --nnodes=1 --master_port=29514 \
  train/train.py \
  --deepspeed scripts/train/zero3.json \
  --stage sft \
  --do_train \
  --model_name_or_path $PRETRAIN_MODEL_PATH \
  --dataset $DATASET \
  --media_dir $MEDIA_DIR \
  --template qwen2_vl \
  --finetuning_type full \
  --output_dir $OUTPUT_PATH \
  --overwrite_cache \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --weight_decay 0.1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --ddp_timeout 90000 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --cutoff_len 4096 \
  --save_steps 10000 \
  --plot_loss \
  --num_train_epochs 1 \
  --bf16 \
  --preprocessing_num_workers 32 \
  2>&1 | tee ${OUTPUT_PATH}/train.log