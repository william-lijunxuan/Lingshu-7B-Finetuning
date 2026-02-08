export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

torchrun --nproc_per_node=2 RL_GRPO_medgemma.py


pip install -U "trl[vllm]"

# check GPU
watch -n 1 nvidia-smi



sudo apt install -y nvtop
nvtop

# 3 GPU
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

accelerate launch RL_GRPO_medgemma.py