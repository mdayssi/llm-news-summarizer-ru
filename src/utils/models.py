def get_max_input_tokens(tokenizer, gen_cfg, reserve=64, default_ctx=2048):
    ctx = getattr(tokenizer, "model_max_length", None)
    if ctx is None or ctx > 100_000_000:
        ctx = default_ctx
    mn = getattr(gen_cfg, "max_new_tokens", 0) or 0
    return max(8, ctx - mn - reserve)

