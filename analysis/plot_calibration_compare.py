import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs("../plots/global", exist_ok=True)

models = ["SLM", "LLM"]

# ECE values
squad = [0.3936, 0.3089]
squad_v2 = [0.7197, 0.654]
hotpot = [0.5735, 0.4275]

x = range(len(models))

plt.figure(figsize=(6,5))

# Improved styling (thicker lines + markers)
plt.plot(x, squad, marker='o', linewidth=2, label="SQuAD")
plt.plot(x, squad_v2, marker='o', linewidth=2, label="SQuAD v2")
plt.plot(x, hotpot, marker='o', linewidth=2, label="HotpotQA")

# Axis settings
plt.xticks(x, models)
plt.ylabel("Expected Calibration Error (ECE)")
plt.title("Calibration Comparison Across Models and Datasets")

# Keep scale consistent
plt.ylim(0, 1)

# Grid + legend
plt.grid(alpha=0.3)
plt.legend()

# Save
plt.savefig("../plots/global/ece_comparison.png", dpi=300, bbox_inches='tight')
plt.close()