from typing import List, Tuple
from razdel import sentenize


def get_max_input_tokens(tokenizer, gen_cfg, reserve=128, default_ctx=32768):
    ctx = getattr(tokenizer, "model_max_length", None)
    if ctx is None or ctx > 100_000_000:
        ctx = default_ctx
    mn = getattr(gen_cfg, "max_new_tokens", 0) or 0
    return max(256, ctx - mn - reserve)

def sample_exemplars(pool_df, k=2, random_state=42) -> List[Tuple[str, str]]:
    ex = pool_df.sample(n=min(k, len(pool_df)), random_state=random_state)
    return list(zip(ex["text"].tolist(), ex["summary"].tolist()))

def assemble_msgs(exs, tgt_text, SYSTEM_PROMPT):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for (x, y) in exs:
        msgs.append({"role": "user", "content": f"Задача: кратко резюмируй.\n\nТекст статьи:\n{x}"})
        msgs.append({"role": "assistant", "content": y})
    msgs.append({"role": "user", "content": f"Задача: кратко резюмируй.\n\nТекст статьи:\n{tgt_text}"})
    return msgs

def lead3(text: str) -> str:
    sents = [s.text.strip() for s in sentenize(text or "")]
    return " ".join(sents[:3])