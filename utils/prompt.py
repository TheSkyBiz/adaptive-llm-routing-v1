def format_prompt(tokenizer, context, question, dataset="squad", model_type="slm"):

    # ----------------------------
    # SYSTEM PROMPT
    # ----------------------------

    if model_type == "slm":
        system_prompt = (
            "Answer using ONLY the context. "
            "If unsure, say: I don't know. "
            "Keep answer short."
        )
    else:  # LLM
        system_prompt = (
            "Carefully read the context and extract the exact answer span. "
            "If multiple pieces are needed, combine them. "
            "Do not guess or infer beyond the context. "
            "If the answer is not present, respond exactly: I don't know. "
            "Return only the final concise answer."
        )   
    # ----------------------------
    # DATASET-SPECIFIC TWEAKS
    # ----------------------------

    if dataset == "squad_v2":
        system_prompt += (
            "Some questions may not have an answer in the context. "
            "If the answer is missing or uncertain, respond exactly: I don't know. "
            "Do not guess or infer beyond the given context."
        )

    elif dataset == "hotpot_qa":
        system_prompt += (
            "The answer may require combining information from multiple parts of the context. "
            "Carefully identify and connect relevant pieces before answering. "
            "Return only the final concise answer without explanation."
        )

    # ----------------------------
    # FINAL MESSAGE
    # ----------------------------

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )