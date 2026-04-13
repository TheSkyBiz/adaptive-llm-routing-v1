import json
import string
import re
from collections import Counter
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt


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
# Chatterjee Correlation
# ----------------------------
def chatterjee_correlation(x, y):
    """
    Computes Chatterjee's rank correlation coefficient ξ
    Measures how strongly x determines y
    """

    x = np.array(x)
    y = np.array(y)

    n = len(x)

    # Rank X
    rx = rankdata(x)

    # Sort Y according to ranked X
    order = np.argsort(rx)
    y_sorted = y[order]

    # Rank Y
    ry = rankdata(y_sorted)

    diff = np.abs(np.diff(ry))

    numerator = np.sum(diff)
    denominator = np.sum(np.abs(ry - np.mean(ry)))

    if denominator == 0:
        return 0

    xi = 1 - (numerator / denominator)

    return xi


# ----------------------------
# Evaluation + Analysis
# ----------------------------
def evaluate(log_path):

    entries = []

    with open(log_path, "r") as f:
        for line in f:
            entry = json.loads(line)

            gold = entry["ground_truth"]
            pred = entry["slm_answer"]

            exact = compute_exact(gold, pred)
            f1 = compute_f1(gold, pred)

            entry["exact_match"] = exact
            entry["f1"] = f1
            entry["correct"] = exact == 1

            # confidence = exp(logprob)
            if entry["slm_avg_logprob"] is not None:
                entry["confidence"] = float(np.exp(entry["slm_avg_logprob"]))
            else:
                entry["confidence"] = 0.0

            entries.append(entry)

    total = len(entries)

    exact_scores = [e["exact_match"] for e in entries]
    f1_scores = [e["f1"] for e in entries]

    print("========== Baseline ==========")
    print(f"Total Samples: {total}")
    print(f"Exact Match: {np.mean(exact_scores):.4f}")
    print(f"F1 Score: {np.mean(f1_scores):.4f}")

    # ----------------------------
    # Extract Signals
    # ----------------------------

    lengths = [e["slm_length"] for e in entries]
    entropies = [e["slm_avg_entropy"] for e in entries]
    logprobs = [e["slm_avg_logprob"] for e in entries]
    times = [e["time_taken_seconds"] for e in entries]
    correct_binary = [1 if e["correct"] else 0 for e in entries]

    # ----------------------------
    # Pearson Correlations
    # ----------------------------
    print("\n========== Pearson Correlation ==========")

    def corr(a, b):
        return np.corrcoef(a, b)[0, 1]

    print(f"Length vs Correct: {corr(lengths, correct_binary):.4f}")
    print(f"Entropy vs Correct: {corr(entropies, correct_binary):.4f}")
    print(f"LogProb vs Correct: {corr(logprobs, correct_binary):.4f}")
    print(f"Time vs Correct: {corr(times, correct_binary):.4f}")

    # ----------------------------
    # Chatterjee Correlation
    # ----------------------------
    print("\n========== Chatterjee Correlation (ξ) ==========")

    length_xi = chatterjee_correlation(lengths, correct_binary)
    entropy_xi = chatterjee_correlation(entropies, correct_binary)
    logprob_xi = chatterjee_correlation(logprobs, correct_binary)
    time_xi = chatterjee_correlation(times, correct_binary)

    print(f"Length → Correct: {length_xi:.4f}")
    print(f"Entropy → Correct: {entropy_xi:.4f}")
    print(f"LogProb → Correct: {logprob_xi:.4f}")
    print(f"Time → Correct: {time_xi:.4f}")

    print("\n========== Length-Based Routing ==========")

    thresholds = [2,4,6,8,10,12]

    for tau in thresholds:

        accepted = [e for e in entries if e["slm_length"] <= tau]
        escalated = [e for e in entries if e["slm_length"] > tau]

        if len(accepted) == 0:
            continue

        accept_rate = len(accepted)/total
        accepted_em = np.mean([e["exact_match"] for e in accepted])
        accepted_f1 = np.mean([e["f1"] for e in accepted])

        print(f"\nThreshold τ = {tau}")
        print(f"Accepted %: {accept_rate:.3f}")
        print(f"Accepted EM: {accepted_em:.4f}")
        print(f"Accepted F1: {accepted_f1:.4f}")

    # ----------------------------
    # Calibration (ECE)
    # ----------------------------
    print("\n========== Calibration (ECE) ==========")

    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)

    ece = 0.0

    for i in range(num_bins):

        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        bin_entries = [
            e for e in entries
            if bin_lower <= e["confidence"] < bin_upper
        ]

        if len(bin_entries) == 0:
            continue

        bin_confidence = np.mean([e["confidence"] for e in bin_entries])
        bin_accuracy = np.mean([e["correct"] for e in bin_entries])

        bin_weight = len(bin_entries) / total
        gap = abs(bin_accuracy - bin_confidence)

        ece += bin_weight * gap

        print(f"\nBin {i+1}: [{bin_lower:.2f}, {bin_upper:.2f})")
        print(f"  Samples: {len(bin_entries)}")
        print(f"  Avg Confidence: {bin_confidence:.4f}")
        print(f"  Accuracy: {bin_accuracy:.4f}")
        print(f"  Gap: {gap:.4f}")

    print("\nFinal ECE:", round(ece, 4))


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":

    evaluate("logs/hotpot_qa_slm.jsonl")