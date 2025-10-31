import os
import sys
import torch
import random
import numpy as np
from tqdm import tqdm
import json
from argparse import ArgumentParser
from distutils.util import strtobool


# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
#
# from utils.Logger_util import Logger

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

from qwenvl.eval.datasets.utils.Logger_util import Logger
from VLM.VLMs import init_llm

import gc
import importlib



parser = ArgumentParser()
parser.add_argument(
    "--config_model_name",
    type=str,
    default="Lingshu",
    help="name of model, e.g. Lingshu, MedVLM_R1, medgemma_4b_it"
)
pre_args, _ = parser.parse_known_args()

repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    config = importlib.import_module(f"config.config_{pre_args.config_model_name}")
    print(f"Loaded config module: config_{pre_args.config_model_name}")
except ModuleNotFoundError:
    config = importlib.import_module("model.config.config")
    print("Fallback to default config module: config")

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def parse_eval_datasets(datasets_str):
    return datasets_str.split(',')



parser.add_argument('--eval_local_datasets_flag', type=str, default=config.EVAL_LOCAL_DATASETS_FLAG,
                    help='flag of eval local dataset')
parser.add_argument('--eval_local_datasets_file', type=parse_eval_datasets, default=config.EVAL_LOCAL_DATASETS_FILE,
                    help='file of eval dataset')
parser.add_argument('--eval_datasets', type=parse_eval_datasets, default=config.EVAL_DATASETS,
                    help='name of eval dataset')
parser.add_argument('--datasets_path', type=str, default=config.DATASETS_PATH)

parser.add_argument('--output_path', type=str, default=config.OUTPUT_PATH,
                    help='name of save json')

parser.add_argument('--model_name', type=str, default=config.MODEL_NAME,
                    help='name of model')
parser.add_argument('--model_path', type=str, default=config.MODEL_PATH)

# parser.add_argument('--eval_dataset_path', type=str, default=config.EVAL_DATASET_PATH)
# parser.add_argument('--eval_output_path', type=str, default=config.EVAL_OUTPUT_PATH)

parser.add_argument('--seed', type=int, default=config.SEED)
parser.add_argument('--cuda_visible_devices', type=str, default=config.CUDA_VISIBLE_DEVICES)
parser.add_argument('--tensor_parallel_size', type=str, default=config.TENSOR_PARALLEL_SIZE)
parser.add_argument('--use_vllm', type=str, default=config.USE_VLLM)
parser.add_argument('--reasoning', type=str, default=config.REASONING)

parser.add_argument('--num_chunks', type=str, default="1")
parser.add_argument('--chunk_idx', type=str, default="0")

parser.add_argument('--max_image_num', type=int, default=1)
parser.add_argument('--max_new_tokens', type=int, default=config.MAX_NEW_TOKENS)
parser.add_argument('--temperature', type=float, default=config.TEMPERATURE)
parser.add_argument('--top_p', type=float, default=config.TOP_P)
parser.add_argument('--repetition_penalty', type=float, default=config.REPETITION_PENALTY)

parser.add_argument('--test_times', type=int, default=config.TEST_TIMES)

parser.add_argument('--use_llm_judge', type=str, default="False")
parser.add_argument('--judge_gpt_model', type=str, default="None")

parser.add_argument('--openai_api_key', type=str, default="rqwerqwerqwerqwrqw")

# RAG
parser.add_argument('--rag_flag', type=str, default=config.RAG_FLAG)
parser.add_argument('--chroma_persist_path', type=str, default=config.CHROMA_PERSIST_PATH)
parser.add_argument('--chroma_collection_name', type=str, default=config.CHROMA_COLLECTION_NAME)
parser.add_argument('--embedding_model_name', type=str, default=config.EMBEDDING_MODEL_NAME)
parser.add_argument('--image_num', type=str, required=False,default="0")
parser.add_argument('--adapter_path', type=str, required=False,default=config.ADAPTER_PATH)

args = parser.parse_args()

os.environ["VLLM_USE_V1"] = "0"



# llm judge setting
if args.openai_api_key == "None" and args.use_llm_judge == "True":
    raise ValueError("If you want to use llm judge, please set the openai api key")

os.environ["judge_gpt_model"] = args.judge_gpt_model
os.environ["use_llm_judge"] = args.use_llm_judge
os.environ["openai_api_key"] = args.openai_api_key

os.environ["eval_local_datasets_flag"] = args.eval_local_datasets_flag
eval_local_datasets_file = ",".join(args.eval_local_datasets_file)
os.environ["eval_local_datasets_file"] = eval_local_datasets_file

# RAG
os.environ["rag_flag"] = args.rag_flag
os.environ["chroma_persist_path"] = args.chroma_persist_path
os.environ["chroma_collection_name"] = args.chroma_collection_name
os.environ["embedding_model_name"] = args.embedding_model_name
os.environ["image_num"] = args.image_num

# eval modle setting
os.environ["REASONING"] = args.reasoning
os.environ["use_vllm"] = args.use_vllm

os.environ["max_image_num"] = str(args.max_image_num)

# vllm and torch setting
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
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


print(f"RAG:{config.DATASET_NAME}\n")
print(f"initializing {config.MODEL_NAME} VLM...")

model_name = args.model_name
MODEL_PATH = args.model_path
eval_dataset = args.datasets_path


logger = Logger(model_name)


try:
    print(f"RAG:{config.RAG_FLAG}\n")
    if bool(strtobool(config.RAG_FLAG)):
        print(f"RAG:{config.DATASET_NAME}\n")

    print(f"initializing {config.MODEL_NAME} VLM...")
    model = init_llm(args)

    total_results_path = os.path.join(args.output_path, 'total_results.json')

    for idx, eval_dataset in enumerate(tqdm(args.eval_datasets), start=0):
        set_seed(args.seed)
        print(f'evaluating on {eval_dataset}...')

        eval_dataset_path = os.path.join(args.datasets_path, eval_dataset) if args.datasets_path != "hf" else None
        if bool(strtobool(config.RAG_FLAG)):
            eval_output_path = os.path.join(args.output_path, eval_dataset + str(idx)+"RAG")
        else:
            eval_output_path = os.path.join(args.output_path, eval_dataset + str(idx)+"Baseline")

        os.makedirs(eval_output_path, exist_ok=True)
        from qwenvl.eval.benchmark import prepare_benchmark

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
    with open(os.path.join(args.output_path, 'total_results.json'), 'w') as f:
        json.dump(total_results, f, indent=4)


except Exception as e:

    import traceback
    traceback.print_exc()


finally:

    logger.close()