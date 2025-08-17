from typing import Optional, List, Dict

import evaluate
import torch
import pandas as pd
from razdel import sentenize
from statistics import mean
from transformers.models.auto.video_processing_auto import model_type


def clean(s: pd.Series) -> pd.Series:
    s = s.fillna("")
    s = (
        s.str.replace("\xa0", " ", regex=False)
        .str.replace("\u2009", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return s


def to_rougeLsum_text(text: Optional[str]) -> Optional[str]:
    return "\n".join(s.text.strip() for s in sentenize(text))


def get_rouge_f1(preds: List[str], refs: List[str]) -> Dict[str, float]:
    metric = evaluate.load("rouge")

    preds_proc = [to_rougeLsum_text(p) for p in preds]
    refs_proc = [to_rougeLsum_text(r) for r in refs]

    scores = metric.compute(
        predictions=preds_proc,
        references=refs_proc,
        use_stemmer=True,
    )
    return scores


def get_bertscorepreds(preds: List[str], refs: List[str], bert_model="xlm-roberta-base", device=None, batch_size=8) -> Dict[str, float]:
    metric = evaluate.load("bertscore")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    bs = metric.compute(
        predictions=preds,
        references=refs,
        model_type=bert_model,
        lang="ru",
        rescale_with_baseline=False,
        batch_size=batch_size,
    )

    scores = {
        "bertscore_precision": mean(bs["precision"]),
        "bertscore_recall": mean(bs["recall"]),
        "bertscore_f1": mean(bs["f1"]),
    }

    return scores

def get_avglen(preds: List[str], refs: List[str]) -> Dict[str, float]:
    avg_len_pred = mean(len(p.split()) for p in preds)
    avg_len_ref = mean(len(r.split()) for r in refs)
    scores = {
        "avg_len_pred": avg_len_pred,
        "avg_len_ref": avg_len_ref,
        "len_ratio_pred_to_ref": (avg_len_pred / avg_len_ref) if avg_len_ref else None
    }

    return scores

def get_all_scores(preds: List[str], refs: List[str], bert_model="xlm-roberta-base", device=None, batch_size=8) -> Dict[str, float]:

    rouge_scores = get_rouge_f1(preds=preds, refs=refs)
    bert_scores = get_bertscorepreds(preds=preds, refs=refs, bert_model=bert_model, device=device, batch_size=batch_size)
    avg_len = get_avglen(preds=preds, refs=refs)

    return {**rouge_scores, **bert_scores, **avg_len}