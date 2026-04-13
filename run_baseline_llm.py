import json
import yaml
import random
import numpy as np
import os
import time
from datasets import load_dataset
from tqdm import tqdm

from models.llm_model import LLMModel
from utils.prompt import format_prompt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------
# Clean LLM output (ADDED)
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


def main():

    config = load_config()
    set_seed(config["run"]["seed"])
    os.makedirs("logs", exist_ok=True)

    print("Loading LLM model...")
    llm = LLMModel()
    print("Model loaded.\n")

    # ----------------------------
    # Dataset loading
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

    output_path = f"logs/{dataset_name}_llm.jsonl"

    with open(output_path, "w") as log_file:

        for example in tqdm(split):

            # ----------------------------
            # Dataset-specific parsing
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
            # Prompt
            # ----------------------------
            prompt = format_prompt(
                tokenizer=llm.tokenizer,
                context=context,
                question=question,
                dataset=dataset_name,
                model_type="llm"
            )

            start_time = time.time()

            raw_answer, avg_logprob, avg_entropy = llm.generate(
                prompt,
                max_new_tokens=config["generation"]["llm_max_new_tokens"],
                temperature=config["generation"]["temperature"],
                top_p=config["generation"]["top_p"]
            )

            # ----------------------------
            # CLEANING (ADDED)
            # ----------------------------
            cleaned_answer = clean_llm_output(raw_answer)

            duration = time.time() - start_time

            log_entry = {
                "id": question_id,
                "question": question,
                "ground_truth": ground_truth,

                "llm_raw_answer": raw_answer,
                "llm_answer": cleaned_answer,

                "llm_length": len(cleaned_answer.split()),
                "llm_avg_logprob": avg_logprob,
                "llm_avg_entropy": avg_entropy,
                "time_taken_seconds": duration,
                "correct": None
            }

            log_file.write(json.dumps(log_entry) + "\n")

    print("LLM baseline run complete.")


if __name__ == "__main__":
    main()