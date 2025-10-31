#!/bin/bash
today_str=$(date +%Y%m%d_%H%M%S)
echo $today_str

source ~/miniforge3/etc/profile.d/conda.sh
conda activate jupyter_env

# dataset and RAG config
DATASET_NAME="MMSkinQA_SKINgpt"
RAG_FLAG="False"
PROJECT_ROOT="/home/william/model/Skinalor/RAG/RAGDataSet"
DB_DIR="${PROJECT_ROOT}/${DATASET_NAME}"
CHROMA_PERSIST_PATH="${DB_DIR}/chroma_db_skin"
CHROMA_COLLECTION_NAME="skin_cases_multivector_${DATASET_NAME}"

# embedding model config
EMBEDDING_MODEL_NAME="openai/clip-vit-base-patch32"

# dataset config
DATASETS_PATH="redlessone"
EVAL_DATASETS="Derm1m"
EVAL_LOCAL_DATASETS_FLAG="True"
EVAL_LOCAL_DATASETS_FILE="/home/william/dataset/skin/Derm1M/Derm1M_train.jsonl"

# output config
OUTPUT_PATH="eval_results/Lingshu-7B"

# VLM model config
MODEL_PATH="/home/william/model/Lingshu-7B"
MODEL_NAME="LingShu"
CONFIG_MODEL_NAME="Lingshu"
ADAPTER_PATH="/home/william/model/Lingshu-7B-Finetuning/qwenvl/scripts/output"

# vllm settings
CUDA_VISIBLE_DEVICES="0"
TENSOR_PARALLEL_SIZE="1"
USE_VLLM="True"

# evaluation settings
SEED=42
REASONING="False"
TEST_TIMES=1
MAX_NEW_TOKENS=1024
MAX_IMAGE_NUM=6
TEMPERATURE=0.7
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge settings
USE_LLM_JUDGE="False"
GPT_MODEL="gpt-4.1-2025-04-14"
OPENAI_API_KEY=""

# run evaluation
python eval_sh.py \
  --config_model_name "$CONFIG_MODEL_NAME" \
  --eval_local_datasets_flag "$EVAL_LOCAL_DATASETS_FLAG" \
  --eval_local_datasets_file "$EVAL_LOCAL_DATASETS_FILE" \
  --eval_datasets "$EVAL_DATASETS" \
  --datasets_path "$DATASETS_PATH" \
  --output_path "$OUTPUT_PATH/$today_str" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --seed "$SEED" \
  --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --use_vllm "$USE_VLLM" \
  --reasoning "$REASONING" \
  --num_chunks 1 \
  --chunk_idx 0 \
  --max_image_num "$MAX_IMAGE_NUM" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --repetition_penalty "$REPETITION_PENALTY" \
  --test_times "$TEST_TIMES" \
  --use_llm_judge "$USE_LLM_JUDGE" \
  --judge_gpt_model "$GPT_MODEL" \
  --openai_api_key "$OPENAI_API_KEY" \
  --rag_flag "$RAG_FLAG" \
  --dataset_name "$DATASET_NAME" \
  --chroma_persist_path "$CHROMA_PERSIST_PATH" \
  --chroma_collection_name "$CHROMA_COLLECTION_NAME" \
  --embedding_model_name "$EMBEDDING_MODEL_NAME" \
  --adapter_path "$ADAPTER_PATH" \
