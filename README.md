# Lingshu-7B-Finetuning
VLM finetuning

## Download code
```bash
git clone https://github.com/william-lijunxuan/Lingshu-7B-Finetuning.git
```
## Create conda env

```bash
cd Lingshu-7B-Finetuning
conda env update -f environment.yml --prune
conda activate lingshu
cd qwenvl/train
python run_sft.py
```

## Download model
Download it to the **model** directory.
```bash
git clone https://huggingface.co/lingshu-medical-mllm/Lingshu-7B
```

## Download dataset
Download it to the **dataset** directory.
```bash
git clone https://huggingface.co/datasets/redlessone/Derm1M
cd Derm1M
# Unzip the file
for file in *.zip; do
    unzip -o "$file" -d "${file%.zip}"
done
#unzip validation_data.zip -d validation_data
#unzip IIYI.zip -d IIYI
#unzip edu.zip -d edu
#unzip note.zip -d note
#unzip public.zip -d public
#unzip pubmed.zip -d pubmed
#unzip reddit.zip -d reddit
#unzip twitter.zip -d twitter
#unzip youtube.zip -d youtube

```
##  Login wandb
```bash
wandb
mr.william.ljx@gmail.com 
1be3c3080c7714f2f5e1c1fb9e78ec54bdbc0193
```

## Learning resources
### peft
https://huggingface.co/docs/peft/en/index
### LoRA
https://github.com/microsoft/LoRA.git
### SFT、RLHF、DPO
Supervised Fine-Tuning

Reinforcement Learning from Human Feedback

Direct Preference Optimization

https://www.bilibili.com/video/BV14whCzfE1R/?spm_id_from=333.1387.collection.video_card.click&vd_source=dedf6783106f532e3945a6712d4876bf

### RAG/Fine Tuning
https://www.bilibili.com/video/BV1AqhAzCEgi?spm_id_from=333.788.videopod.sections&vd_source=dedf6783106f532e3945a6712d4876bf


### Fine tuning tools
https://github.com/hiyouga/LLaMA-Factory.git

https://github.com/unslothai/unsloth.git

https://github.com/modelscope/ms-swift

https://github.com/hpcaitech/ColossalAI

### Evaluation
https://github.com/alibaba-damo-academy/MedEvalKit.git

https://github.com/modelscope/evalscope.git