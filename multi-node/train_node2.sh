# export NCCL_SOCKET_IFNAME=eth1
# export NCCL_IB_DISABLE=1
nnodes=2
nproc_per_node=4

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=$nnodes \
NODE_RANK=1 \
MASTER_ADDR=11.204.14.164 \
MASTER_PORT=29500 \
NPROC_PER_NODE=$nproc_per_node \
swift sft --model  /path/to/Qwen2.5-VL-3B-Instruct  --num_train_epochs 3  --learning_rate 5e-5   --warmup_ratio 0.03      --per_device_train_batch_size 2  --per_device_eval_batch_size 2   --gradient_accumulation_steps 2   --train_type lora  --lora_rank 16   --lora_alpha 32  --freeze_vit true   --max_length 4096 --dataset /path/to/all_data_0212.jsonl --output_dir /path/to/work_dirs/qwen2_vl    --save_total_limit 10   --seed 0   --eval_strategy no --save_steps 500 --val_dataset /path/to/aigc_tongyong_detect_swift_bal_gan_val.jsonl

