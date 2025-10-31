import os
from  datetime import datetime

today_str  = datetime.now().strftime("%Y%m%d_%H%M%S")

# RAG config --MMSkinQA_SkinCAP
# DATASET_NAME ="MMSkinQA_SkinCAP"
# DATASET_NAME ="MMSkinQA"
DATASET_NAME ="MMSkinQA_SKINgpt"
# DATASET_NAME ="SkinCAP"
RAG_FLAG="True"
PROJECT_ROOT = "/home/william/model/Skinalor/RAG/RAGDataSet"
DB_DIR = os.path.join(PROJECT_ROOT, DATASET_NAME)
CHROMA_PERSIST_PATH = os.path.join(DB_DIR, "chroma_db_skin")
CHROMA_COLLECTION_NAME = r"skin_cases_multivector_"+DATASET_NAME


# RAG_FLAG="True"
# # DATASET_NAME ="SkinCAP"
# DATASET_NAME ="MMSkinQA_SKINgpt"
# PROJECT_ROOT = "/home/william/model/Skinalor/RAG/RAGDataSet"
# DB_DIR = os.path.join(PROJECT_ROOT, DATASET_NAME)
# CHROMA_PERSIST_PATH = os.path.join(DB_DIR, "chroma_db_skin")
# CHROMA_COLLECTION_NAME = "skin_cases_multivector"+DATASET_NAME

# embedding model config
EMBEDDING_MODEL_NAME = 'openai/clip-vit-base-patch32'

# dataSet
DATASETS_PATH = "HemanthKumarK"
# EVAL_DATASETS = "SkinCAP,SkinCAP,SKINgpt,MMSkinQA"
EVAL_DATASETS = "SkinCAP"
# EVAL_DATASETS = "MMSkinQA"


EVAL_LOCAL_DATASETS_FLAG ="True"
# EVAL_LOCAL_DATASETS_FILE ="/home/william/dataset/skin/SkinCAP/SkinCAP_20250712_121252.json,/home/william/dataset/skin/SkinCAP/SkinCAP_20250712_013256.json,/home/william/dataset/skin/SKINgpt/20250711055029_SKINgpt_multiple_choice_QA.json,/home/william/dataset/skin/MM-SkinQA/MM-SkinQA_20250711213519.json"
# EVAL_LOCAL_DATASETS_FILE ="/home/william/dataset/skin/SKINgpt/20250717055029_SKINgpt_close_end_QA.json"
# EVAL_LOCAL_DATASETS_FILE ="/home/william/dataset/skin/SkinCAP/SkinCAP_20250712_121252.json"
# EVAL_LOCAL_DATASETS_FILE ="/home/william/dataset/skin/SkinCAP/SkinCAP_20250717_201435_multiple_choice_QA.json"
EVAL_LOCAL_DATASETS_FILE ="/home/william/dataset/skin/SkinCAP/SkinCAP_20250712_121252_close_end_QA.json"


# EVAL_DATASET_PATH = "/home/william/dataset/skin/SKINgpt"
OUTPUT_PATH = f"eval_results/medgemma-4b-it/{today_str}"

# VLM model path
MODEL_PATH = "/home/william/model/medgemma-4b-it"
MODEL_NAME = "MedGemma"


#vllm setting
CUDA_VISIBLE_DEVICES="0"
TENSOR_PARALLEL_SIZE="1"
USE_VLLM="True"

#Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1


# Eval LLM setting
MAX_NEW_TOKENS=1024
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge setting
USE_LLM_JUDGE="False"
# gpt api model name
GPT_MODEL="gpt-4.1-2025-04-14"
OPENAI_API_KEY=""



