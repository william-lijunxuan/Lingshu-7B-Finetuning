export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

torchrun --nproc_per_node=2 RL_GRPO_medgemma.py


pip install -U "trl[vllm]"

# check GPU
watch -n 1 nvidia-smi



sudo apt install -y nvtop
nvtop