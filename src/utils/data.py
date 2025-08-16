from typing import Optional, List, Dict

import evaluate
import pandas as pd
from razdel import sentenize


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
    refs_proc  = [to_rougeLsum_text(r) for r in refs]

    scores = metric.compute(
        predictions=preds_proc,
        references=refs_proc,
        use_stemmer=False,
    )
    return scores