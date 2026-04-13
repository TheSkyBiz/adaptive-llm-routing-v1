import json
import string
import re
from collections import Counter
import numpy as np
import yaml


# ----------------------------
# Config Loader
# ----------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------
# Normalization
# ----------------------------
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ----------------------------
# Metrics
# ----------------------------
def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):

    gold_tokens = normalize_answer(a_gold).split()
    pred_tokens = normalize_answer(a_pred).split()

    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())

    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return int(gold_tokens == pred_tokens)

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_routing(log_path):

    entries = []

    with open(log_path, "r") as f:
        for line in f:
            entry = json.loads(line)

            gold = entry["ground_truth"]
            pred = entry["final_answer"]

            # 🔥 Clean prediction FIRST
            pred = pred.strip().lower()

            # Normalize after cleaning
            pred_norm = normalize_answer(pred)
            gold_norm = normalize_answer(gold)

            # ----------------------------
            # SQuAD v2 handling
            # ----------------------------
            if gold_norm == "":
                # correct if model abstains
                if pred_norm in ["i dont know", "i dontknow"]:
                    exact = 1
                    f1 = 1
                else:
                    exact = 0
                    f1 = 0
            else:
                exact = compute_exact(gold, pred)
                f1 = compute_f1(gold, pred)

            entry["exact_match"] = exact
            entry["f1"] = f1

            entries.append(entry)

    total = len(entries)

    exact_scores = [e["exact_match"] for e in entries]
    f1_scores = [e["f1"] for e in entries]

    print("========== ROUTING RESULTS ==========")
    print(f"Total Samples: {total}")
    print(f"Final Exact Match: {np.mean(exact_scores):.4f}")
    print(f"Final F1 Score: {np.mean(f1_scores):.4f}")

    # ----------------------------
    # Routing stats
    # ----------------------------
    routed_to_llm = [e for e in entries if e["routed_to"] == "LLM"]
    routed_to_slm = [e for e in entries if e["routed_to"] == "SLM"]

    llm_rate = len(routed_to_llm) / total
    slm_rate = len(routed_to_slm) / total

    print("\n========== ROUTING STATS ==========")
    print(f"SLM handled %: {slm_rate:.3f}")
    print(f"LLM handled %: {llm_rate:.3f}")

    # ----------------------------
    # Accuracy split
    # ----------------------------
    if len(routed_to_slm) > 0:
        slm_em = np.mean([e["exact_match"] for e in routed_to_slm])
        print(f"\nSLM Accuracy (accepted): {slm_em:.4f}")

    if len(routed_to_llm) > 0:
        llm_em = np.mean([e["exact_match"] for e in routed_to_llm])
        print(f"LLM Accuracy (escalated): {llm_em:.4f}")

    # ----------------------------
    # Cost Simulation
    # ----------------------------
    SLM_COST = 1
    LLM_COST = 10

    total_cost = (
        len(routed_to_slm) * SLM_COST +
        len(routed_to_llm) * LLM_COST
    )

    avg_cost = total_cost / total

    print("\n========== COST ANALYSIS ==========")
    print(f"Total Cost: {total_cost}")
    print(f"Average Cost per Query: {avg_cost:.2f}")

    # ----------------------------
    # Latency
    # ----------------------------
    avg_time = np.mean([e["time_taken_seconds"] for e in entries])

    print("\n========== LATENCY ==========")
    print(f"Average Time per Query: {avg_time:.3f} sec")

    # ----------------------------
    # Error Analysis
    # ----------------------------
    wrong_answers = [e for e in entries if e["exact_match"] == 0]

    print("\n========== ERROR ANALYSIS ==========")
    print(f"Error Rate: {len(wrong_answers)/total:.4f}")

    # ----------------------------
    # Summary
    # ----------------------------
    print("\n========== SUMMARY ==========")
    print("Accuracy vs Cost trade-off achieved.")
    print(f"LLM usage reduced to {llm_rate:.2%} of queries.")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":

    config = load_config()

    mode = config["routing"]["mode"]

    log_path = f"logs/hotpot_qa_routing_{mode}.jsonl"

    print(f"Evaluating routing mode: {mode}")
    print(f"Using log file: {log_path}\n")

    evaluate_routing(log_path)