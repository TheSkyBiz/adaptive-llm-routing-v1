import json
import yaml
import random
import numpy as np
import os
import time
from datasets import load_dataset
from tqdm import tqdm

from models.slm_model import SLMModel
from utils.prompt import format_prompt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    config = load_config()
    set_seed(config["run"]["seed"])
    os.makedirs("logs", exist_ok=True)

    print("Loading SLM model...")
    slm = SLMModel()
    print("Model loaded.\n")

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

    output_path = f"logs/{dataset_name}_slm.jsonl"

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
            # Prompt (UPDATED)
            # ----------------------------
            prompt = format_prompt(
                tokenizer=slm.tokenizer,
                context=context,
                question=question,
                dataset=dataset_name,
                model_type="slm"
            )

            start_time = time.time()

            answer, avg_logprob, avg_entropy = slm.generate(
                prompt,
                max_new_tokens=config["generation"]["slm_max_new_tokens"],
                temperature=config["generation"]["temperature"],
                top_p=config["generation"]["top_p"]
            )

            duration = time.time() - start_time

            log_entry = {
                "id": question_id,
                "question": question,
                "ground_truth": ground_truth,
                "slm_answer": answer,
                "slm_length": len(answer.split()),
                "slm_avg_logprob": avg_logprob,
                "slm_avg_entropy": avg_entropy,
                "time_taken_seconds": duration,
                "correct": None
            }

            log_file.write(json.dumps(log_entry) + "\n")

    print("SLM baseline run complete.")


if __name__ == "__main__":
    main()