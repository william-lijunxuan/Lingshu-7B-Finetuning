from qwenvl.eval.datasets.utils.base_dataset import BaseDataset


from datasets import load_dataset,Dataset
from tqdm import tqdm
import os
import json
from PIL import Image
from mathruler.grader import extract_boxed_content
from qwenvl.eval.datasets.utils.utils import load_and_maybe_compress,save_json,extract,judger,get_compare_messages,judge_open_end_vqa,judge_judgement,judge_judgement_close_options,judge_close_end_vqa
from qwenvl.eval.datasets.utils.question_formats import get_close_ended_prompt
from distutils.util import strtobool

class Derm1m(BaseDataset):
    def __init__(self,idx,model,dataset_path,output_path):
        self.model = model
        self.idx = idx
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "redlessone/Derm1M"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
        self.eval_local_datasets_flag = bool(strtobool(os.environ.get("eval_local_datasets_flag",True)))
        self.eval_local_datasets_file = str(os.environ.get("eval_local_datasets_file", "/home/william/dataset/skin/Derm1M/Derm1M_train.jsonl"))


    def load_data(self):
        # load local datasets
        if self.eval_local_datasets_flag:
            path = self.eval_local_datasets_file.split(',')[self.idx].strip()
            if path.endswith((".jsonl", ".ndjson")):
                with open(path, "r", encoding="utf-8") as f:
                    records = [json.loads(line) for line in f if line.strip()]
            else:
                with open(path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            train_ds = Dataset.from_list(records)
            dataset = train_ds.select(range(1))
            # dataset = train_ds
            for idx,sample in tqdm(enumerate(dataset)):
                if idx % self.num_chunks == self.chunk_idx:
                    sample = self.construct_multi_image_rag_prompt(sample)
                    self.samples.append(sample)
            return self.samples

    def construct_multi_image_rag_prompt(self,sample):
        prompt_text = """You are a board‐certified dermatology AI specialist. A patient has just uploaded an image of a skin lesion. Carefully examine the lesion’s visual features—color, shape, borders, surface texture, and anatomic location—and then compose a single, fully descriptive diagnostic sentence in English. Mirror the expert style by:
        1. Opening with a concise description of the key visual finding (e.g. “The red, smooth, exophytic nodule with a slightly narrowed base…”).
        2. Stating the most likely diagnosis (e.g. “…may indicate squamous cell carcinoma.”).
        3. Optionally noting any next steps for confirmation (e.g. “Further biopsy is recommended to confirm the diagnosis.”).
        Example output (for a smooth red papule on the lip):
        “The red, smooth, dome-shaped papule on the lip, with slight keratosis and prominent capillaries, is most consistent with basal cell carcinoma; a skin biopsy is advised for confirmation.”"""

        primary_img_path = os.path.join("/home/william/dataset/skin/Derm1M", sample["image"])
        image = Image.open(primary_img_path).convert("RGB")
        messages = {"prompt": prompt_text, "image": image}
        sample["messages"] = messages
        del sample["image"]

        return sample


    def cal_metrics(self,out_samples):
        messages_list = []

        metrics = {
            "total metrics": {
                "total": 0,
                "right": 0
            },
            "open": {
                "total": 0,
                "right": 0,
                "bleu1": 0,
                "bleu2": 0,
                "bleu3": 0,
                "bleu4": 0,
                "rouge1": 0,
                "rouge2": 0,
                "rougel": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "em": 0,
            },
            "close": {
                "total": 0,
                "right": 0
            }
        }

        open_id = []
        for i, out_sample in tqdm(enumerate(out_samples)):
            response = out_sample["response"]
            print(f"response:{response}")
            if extract_boxed_content(response) != "None":
                response = extract_boxed_content(response)
            elif "<answer>" in response:
                response = extract(response, "answer")

            # answer = out_sample["answer"]
            answer = None
            for conv in out_sample["conversations"]:
                if conv["from"] == "gpt":
                    answer = conv["value"]
                    break
                    # question_type = out_sample["question_type"]
            question_type = "open_end_QA"

            answer = answer.lower().strip()
            response = response.lower().strip()

            metrics["total metrics"]["total"] += 1
            if question_type =="multiple_choice_QA":
                metrics["close"]["total"] += 1
                correct = judge_judgement_close_options(answer, response)
                out_samples[i]["correct"] = correct
                if correct:
                    metrics["close"]["right"] += 1
                    metrics["total metrics"]["right"] += 1
            elif question_type == "open_end_QA":
                metrics["open"]["total"] += 1

                c_metrics = judge_open_end_vqa(answer, response)
                out_samples[i]["correct"] = c_metrics["em"]
                out_samples[i]["metrics"] = c_metrics
                if c_metrics["em"]:
                    metrics["total metrics"]["right"] += 1
                    metrics["open"]["right"] += 1
                for metric in c_metrics:
                    metrics["open"][metric] += c_metrics[metric]
            elif question_type == "close_end_QA":
                metrics["close"]["total"] += 1
                correct = judge_close_end_vqa(answer, response)
                out_samples[i]["correct"] = correct
                if correct:
                    metrics["close"]["right"] += 1
                    metrics["total metrics"]["right"] += 1

                # if os.environ.get("use_llm_judge", "False") == "True":
                #     messages = get_compare_messages(question, response, answer)
                #     messages_list.append(messages)
                #     open_id.append(i)

        if os.environ.get("use_llm_judge", "False") == "True":
            metrics["total metrics"]["right"] = 0
            metrics["open"]["right"] = 0
            metrics["close"]["right"] = 0
            llm = judger
            results = llm.generate_outputs(messages_list)
            for i, result in zip(open_id, results):
                result = extract(result, "judge")
                result = True if result == "0" else False
                out_samples[i]["correct"] = result
                if result:
                    metrics["open"]["right"] += 1
                    metrics["total metrics"]["right"] += 1

        total_total = metrics["total metrics"]["total"]
        if total_total > 0:
           metrics["total metrics"]["acc"] = metrics["total metrics"]["right"] / total_total
        else:
            metrics["total metrics"]["acc"] = 0.0

        open_total = metrics["open"]["total"]
        if open_total > 0:
            metrics["open"]["acc"] = metrics["open"]["right"] / open_total
        else:
            metrics["open"]["acc"] = 0.0

        close_total = metrics["close"]["total"]
        if close_total > 0:
            metrics["close"]["acc"] = metrics["close"]["right"] / close_total
        else:
            metrics["close"]["acc"] = 0.0
        if open_total > 0:
            for metric in metrics["open"]:
                if metric not in ["right", "total", "acc"]:
                    metrics["open"][metric] = metrics["open"][metric] / open_total
        else:
            for metric in metrics["open"]:
                if metric not in ["right", "total", "acc"]:
                    metrics["open"][metric] = 0.0

        total_time = float(os.environ.get("total_time"))
        total_samples = metrics["total metrics"]["total"]
        avg_time = total_time / total_samples if total_samples > 0 else 0.0


        metrics["total_time_s"] = total_time
        metrics["avg_time_per_sample_s"] = avg_time
        return metrics, out_samples
