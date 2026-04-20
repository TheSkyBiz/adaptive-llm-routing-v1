import json
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import string


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


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def load_entries(path):
    entries = []
    with open(path) as f:
        for line in f:
            e = json.loads(line)

            gold = e["ground_truth"]
            pred = e["llm_answer"]

            e["correct"] = compute_exact(gold, pred)

            if e["llm_avg_logprob"] is not None:
                e["confidence"] = float(np.exp(e["llm_avg_logprob"]))
            else:
                e["confidence"] = 0.0

            entries.append(e)

    return entries


def get_output_dir(path):
    base = "../plots"
    if "squad_v2" in path:
        return f"{base}/squad_v2"
    elif "hotpot_qa" in path:
        return f"{base}/hotpot_qa"
    return f"{base}/squad"


# ----------------------------
# SAME PLOTS AS SLM
# ----------------------------
def reliability(entries, save):
    bins = np.linspace(0,1,11)
    accs, confs = [], []

    for i in range(len(bins)-1):
        b = [e for e in entries if bins[i] <= e["confidence"] < bins[i+1]]
        if not b:
            continue

        accs.append(np.mean([e["correct"] for e in b]))
        confs.append(np.mean([e["confidence"] for e in b]))

    plt.figure()
    plt.plot(confs, accs, marker='o', linewidth=2)
    plt.plot([0,1],[0,1],'--')
    plt.title("Reliability Diagram (LLM)")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(save, dpi=300)
    plt.close()


def histogram(entries, save):
    plt.figure()
    plt.hist([e["confidence"] for e in entries], bins=20)
    plt.title("Confidence Distribution (LLM)")
    plt.xlabel("Confidence")
    plt.ylabel("Number of Samples with Confidence in Bin")
    plt.grid()
    plt.savefig(save, dpi=300)
    plt.close()


def length_vs_acc(entries, save):
    plt.figure()
    plt.scatter([e["llm_length"] for e in entries],
                [e["correct"] for e in entries],
                alpha=0.3)
    plt.title("Length vs Accuracy (LLM)")
    plt.grid()
    plt.savefig(save, dpi=300)
    plt.close()


def entropy_distribution(entries, save):
    entropy = [e["llm_avg_entropy"] for e in entries if e["llm_avg_entropy"] is not None]

    plt.figure()
    plt.hist(entropy, bins=20)
    plt.title("Entropy Distribution (LLM)")
    plt.grid()
    plt.savefig(save, dpi=300)
    plt.close()


def logprob_distribution(entries, save):
    logprob = [e["llm_avg_logprob"] for e in entries if e["llm_avg_logprob"] is not None]

    plt.figure()
    plt.hist(logprob, bins=20)
    plt.title("LogProb Distribution (LLM)")
    plt.grid()
    plt.savefig(save, dpi=300)
    plt.close()


if __name__ == "__main__":

    log_path = "../logs/squad_llm.jsonl"

    entries = load_entries(log_path)

    out = get_output_dir(log_path)
    os.makedirs(out, exist_ok=True)

    reliability(entries, f"{out}/reliability_llm.png")
    histogram(entries, f"{out}/confidence_llm.png")
    length_vs_acc(entries, f"{out}/length_vs_accuracy_llm.png")
    entropy_distribution(entries, f"{out}/entropy_llm.png")
    logprob_distribution(entries, f"{out}/logprob_llm.png")