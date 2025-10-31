import os
import sys
import torch
import random
import numpy as np
from tqdm import tqdm
import json
from argparse import ArgumentParser
import gc


script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

from qwenvl.eval.datasets.utils.Logger_util import Logger
from VLM.VLMs import init_llm

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_eval_datasets(datasets_str):
    return datasets_str.split(',')

parser = ArgumentParser()

parser.add_argument('--config_model_name', type=str, required=True, help='Model config name (e.g., medgemma_4b_it)')
parser.add_argument('--eval_local_datasets_flag', type=str, required=True)
parser.add_argument('--eval_local_datasets_file', type=parse_eval_datasets, required=True)
parser.add_argument('--eval_datasets', type=parse_eval_datasets, required=True)
parser.add_argument('--datasets_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--cuda_visible_devices', type=str, required=True)
parser.add_argument('--tensor_parallel_size', type=str, required=True)
parser.add_argument('--use_vllm', type=str, required=True)
parser.add_argument('--reasoning', type=str, required=True)
parser.add_argument('--num_chunks', type=str, required=True)
parser.add_argument('--chunk_idx', type=str, required=True)
parser.add_argument('--max_image_num', type=int, required=True)
parser.add_argument('--max_new_tokens', type=int, required=True)
parser.add_argument('--temperature', type=float, required=True)
parser.add_argument('--top_p', type=float, required=True)
parser.add_argument('--repetition_penalty', type=float, required=True)
parser.add_argument('--test_times', type=int, required=True)
parser.add_argument('--use_llm_judge', type=str, required=True)
parser.add_argument('--judge_gpt_model', type=str, required=True)
parser.add_argument('--openai_api_key', type=str, required=True)
parser.add_argument('--rag_flag', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--chroma_persist_path', type=str, required=True)
parser.add_argument('--chroma_collection_name', type=str, required=True)
parser.add_argument('--embedding_model_name', type=str, required=True)
parser.add_argument('--image_num', type=str, required=False,default="0")
parser.add_argument('--adapter_path', type=str, required=False,default="")
args = parser.parse_args()

# Env setup
os.environ["VLLM_USE_V1"] = "0"
os.environ["judge_gpt_model"] = args.judge_gpt_model
os.environ["use_llm_judge"] = args.use_llm_judge
os.environ["openai_api_key"] = args.openai_api_key
os.environ["eval_local_datasets_flag"] = args.eval_local_datasets_flag
os.environ["eval_local_datasets_file"] = ",".join(args.eval_local_datasets_file)
os.environ["rag_flag"] = args.rag_flag
os.environ["chroma_persist_path"] = args.chroma_persist_path
os.environ["chroma_collection_name"] = args.chroma_collection_name
os.environ["embedding_model_name"] = args.embedding_model_name
os.environ["REASONING"] = args.reasoning
os.environ["use_vllm"] = args.use_vllm
os.environ["max_image_num"] = str(args.max_image_num)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["image_num"] = args.image_num

if args.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

if args.use_vllm == "True":
    os.environ["tensor_parallel_size"] = args.tensor_parallel_size
    if int(args.tensor_parallel_size) > 1:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
else:
    torch.multiprocessing.set_start_method('spawn')
    os.environ["num_chunks"] = args.num_chunks
    os.environ["chunk_idx"] = args.chunk_idx

os.makedirs(args.output_path, exist_ok=True)

model = init_llm(args)
logger = Logger(args.model_name)

total_results_path = os.path.join(args.output_path, 'total_results.json')
from qwenvl.eval.benchmark import prepare_benchmark

print(f"RAG:{args.rag_flag}\n")
print(f"RAG:{args.dataset_name}\n")

try:
    for idx, eval_dataset in enumerate(tqdm(args.eval_datasets), start=0):
        set_seed(args.seed)
        print(f'evaluating on {eval_dataset}...')

        eval_dataset_path = os.path.join(args.datasets_path, eval_dataset) if args.datasets_path != "hf" else None
        eval_output_path = os.path.join(
            args.output_path, eval_dataset + str(idx) + ("RAG" if args.rag_flag == "True" else "Baseline")
        )

        os.makedirs(eval_output_path, exist_ok=True)

        benchmark = prepare_benchmark(idx, model, eval_dataset, eval_dataset_path, eval_output_path)
        benchmark.load_data()
        final_results = benchmark.eval() if benchmark else {}

        print(f'final results on {eval_dataset}: {final_results}')

        if final_results is not None:
            if os.path.exists(total_results_path):
                with open(total_results_path, "r") as f:
                    total_results = json.load(f)
            else:
                total_results = {}
            total_results[eval_dataset] = final_results

        gc.collect()

    with open(total_results_path, 'w') as f:
        json.dump(total_results, f, indent=4)

except Exception as e:
    import traceback
    traceback.print_exc()

finally:
    logger.close()
