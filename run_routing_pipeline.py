import json
import yaml
import random
import numpy as np
import os
import time
from datasets import load_dataset
from tqdm import tqdm

from models.slm_model import SLMModel
from models.llm_model import LLMModel
from utils.prompt import format_prompt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ----------------------------
# Clean LLM output
# ----------------------------
def clean_llm_output(text):

    if text is None:
        return ""

    text = text.replace("|<|im_end|>", "").strip()

    # remove chain artifacts
    if "|" in text:
        text = text.split("|")[0]

    # remove "Answer:" prefix safely
    lower_text = text.lower()
    if "answer:" in lower_text:
        idx = lower_text.find("answer:")
        text = text[idx + len("answer:"):]

    # keep only first line
    text = text.split("\n")[0]

    # strip quotes / spaces
    text = text.strip().strip('"').strip("'")

    return text.strip()


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------
# Routing decision
# ----------------------------
def routing_decision(length, entropy, logprob, mode="v1"):

    if mode == "v1":
        if length > 8 or entropy > 1.5 or logprob < -0.6:
            return "escalate"

    elif mode == "v2":
        if length > 5 or entropy > 1.1 or logprob < -0.45:
            return "escalate"

    elif mode == "v3":
        if length > 10 or entropy > 1.8 or logprob < -0.8:
            return "escalate"

    return "accept"


def main():

    config = load_config()
    set_seed(config["run"]["seed"])

    os.makedirs("logs", exist_ok=True)

    print("Loading models...")

    slm = SLMModel()
    llm = LLMModel()

    print("Models loaded.\n")

    # ----------------------------
    # Dataset loading (UPDATED)
    # ----------------------------
    dataset_name = config["dataset"]["name"]

    if dataset_name == "hotpot_qa":
        dataset = load_dataset("hotpot_qa", config["dataset"]["subset"])
    else:
        dataset = load_dataset(dataset_name)

    split = dataset[config["dataset"]["split"]]

    sample_limit = config["dataset"]["sample_limit"]
    if sample_limit is not None:
        split = split.select(range(sample_limit))

    # ----------------------------
    # Log path (UPDATED)
    # ----------------------------
    output_path = f"logs/{dataset_name}_routing_{config['routing']['mode']}.jsonl"
    print(f"Running routing mode: {config['routing']['mode']}")

    with open(output_path, "w") as log_file:

        for example in tqdm(split):

            # ----------------------------
            # Dataset-specific parsing (UPDATED)
            # ----------------------------
            if dataset_name in ["squad", "squad_v2"]:
                question_id = example["id"]
                question = example["question"]
                context = example["context"]

                if len(example["answers"]["text"]) == 0:
                    ground_truth = ""
                else:
                    ground_truth = example["answers"]["text"][0]

            elif dataset_name == "hotpot_qa":
                question_id = example["id"]
                question = example["question"]

                context_sentences = []
                for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
                    context_sentences.extend(sentences)

                context = " ".join(context_sentences)

                ground_truth = example["answer"]

            # ----------------------------
            # Prompts (UPDATED)
            # ----------------------------
            prompt_slm = format_prompt(
                tokenizer=slm.tokenizer,
                context=context,
                question=question,
                dataset=dataset_name,
                model_type="slm"
            )

            start_time = time.time()

            # ----------------------------
            # SLM inference
            # ----------------------------
            slm_answer, logprob, entropy = slm.generate(
                prompt_slm,
                max_new_tokens=config["generation"]["slm_max_new_tokens"],
                temperature=0,
                top_p=1
            )

            slm_answer = slm_answer.strip()
            length = len(slm_answer.split())

            decision = routing_decision(
                length,
                entropy,
                logprob,
                mode=config["routing"]["mode"]
            )

            # ----------------------------
            # Routing
            # ----------------------------
            if decision == "escalate":

                prompt_llm = format_prompt(
                    tokenizer=llm.tokenizer,
                    context=context,
                    question=question,
                    dataset=dataset_name,
                    model_type="llm"
                )

                llm_answer, _, _ = llm.generate(
                    prompt_llm,
                    max_new_tokens=config["generation"]["llm_max_new_tokens"],
                    temperature=0,
                    top_p=1
                )

                cleaned_llm = clean_llm_output(llm_answer)

                final_answer = cleaned_llm
                routed_to = "LLM"

            else:
                final_answer = slm_answer
                llm_answer = None
                cleaned_llm = None
                routed_to = "SLM"

            duration = time.time() - start_time

            # ----------------------------
            # Logging
            # ----------------------------
            log_entry = {

                "id": question_id,
                "question": question,
                "ground_truth": ground_truth,

                "slm_answer": slm_answer,
                "slm_length": length,
                "slm_entropy": entropy,
                "slm_logprob": logprob,

                "llm_raw_answer": llm_answer,
                "llm_cleaned_answer": cleaned_llm,

                "final_answer": final_answer,
                "routed_to": routed_to,

                "time_taken_seconds": duration
            }

            log_file.write(json.dumps(log_entry) + "\n")

    print("Routing experiment complete.")


if __name__ == "__main__":
    main()