import os
import sys
from typing import List, Dict


import sacrebleu
from rouge_score import rouge_scorer
from rouge_score.scoring import BootstrapAggregator
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import TreebankWordTokenizer
import nltk


try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

_tokenizer = TreebankWordTokenizer()


def compute_bleu4(reference: str, candidates: List[str]) -> List[float]:
    # sacrebleu 需要 refs 是 List[List[str]]，每个子列表是一个参考集
    refs = [[reference]]
    scores = []
    for c in candidates:
        bleu = sacrebleu.corpus_bleu([c], refs, force=True, use_effective_order=True)
        # sacrebleu 返回百分数（0~100），换成 0~1
        scores.append(bleu.score / 100.0)
    return scores


def compute_rouge_l(reference: str, candidates: List[str]) -> List[float]:
    # rouge-score 默认返回 precision/recall/fmeasure，这里取 F1
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = []
    for c in candidates:
        score = scorer.score(reference, c)["rougeL"].fmeasure
        vals.append(score)
    return vals


def compute_meteor(reference: str, candidates: List[str]) -> List[float]:
    # NLTK 3.8+ 的 meteor_score 需要传入“已分词”的 tokens；使用 Treebank 分词器避免 punkt_tab 依赖
    ref_tok = _tokenizer.tokenize(reference)
    vals = []
    for c in candidates:
        cand_tok = _tokenizer.tokenize(c)
        vals.append(meteor_score([ref_tok], cand_tok))
    return vals


def compute_bert_score_recall(reference: str, candidates: List[str], model_type: str = "roberta-large"):
    """
    需要联网下载模型；离线环境会抛错，则返回 None。
    """
    try:
        from bert_score import score
        # bert-score 接受列表；返回张量，取 recall
        P, R, F = score(candidates, [reference] * len(candidates),
                        lang="en", model_type=model_type, verbose=False)
        return [float(r) for r in R]
    except Exception as e:
        sys.stderr.write(f"[WARN] BERTScore 计算失败：{e}\n")
        return None


def evaluate_all(reference: str, candidate_dict: Dict[str, str], bert_model: str = "roberta-large"):
    names = list(candidate_dict.keys())
    cands = [candidate_dict[n] for n in names]

    bleu4 = compute_bleu4(reference, cands)
    rougeL = compute_rouge_l(reference, cands)
    meteor = compute_meteor(reference, cands)
    bert_recall = compute_bert_score_recall(reference, cands, model_type=bert_model)

    # 输出
    print(f"{'name':<12}  {'BLEU-4':>8}  {'ROUGE-L':>8}  {'METEOR':>8}  {'BERT-R':>8}")
    print("-" * 52)
    for i, n in enumerate(names):
        b4 = bleu4[i]
        rl = rougeL[i]
        me = meteor[i]
        br = bert_recall[i] if bert_recall is not None else float("nan")
        print(f"{n:<12}  {b4:8.5f}  {rl:8.5f}  {me:8.5f}  {br:8.5f}")


if __name__ == "__main__":
    # ==== 示例数据（可自行替换/改为读取文件）====
    reference = (
        "The user's chief complaint is related to a skin condition they are experiencing, "
        "specifically described as '皮肿淀粉样变' (Amyloidosis of the skin). They have uploaded a photo depicting "
        "their skin condition. Other forum users responded to the post discussing similar instances and treatments. "
        "One user noted they saw an identical case during their advanced studies. Another user inquired about effective "
        "medications for this skin condition."
    )

    candidates = {
        "res1": (
            "demarcated and are interspersed with areas of erythema (redness). The texture of the skin under the scales "
            "seems rough and possibly inflamed. There is no visible ulceration or bleeding, but the severity of the scaling "
            "suggests significant dermatological involvement. The surrounding skin appears normal in color and texture, "
            "highlighting the localized nature of the condition on these legs."
        ),
        "res2": (
            "The image displays the lower legs of an individual, showing extensive skin changes. Both legs exhibit widespread, "
            "thick, silvery-white scales covering the skin, which is characteristic of psoriasis. The scales appear to be "
            "well-demarcated and are interspersed with areas of erythema (redness). The texture of the skin under the scales "
            "seems rough and possibly inflamed. There is no visible ulceration or bleeding, but the severity of the scaling "
            "suggests significant dermatological involvement. The surrounding skin appears normal in color and texture, "
            "highlighting the localized nature of the condition on these legs."
        ),
    }


    evaluate_all(reference, candidates, bert_model="roberta-large")
