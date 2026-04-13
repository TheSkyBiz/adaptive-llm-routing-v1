import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs("../plots/global", exist_ok=True)

datasets = ["SQuAD", "SQuAD v2", "HotpotQA"]
llm_usage = [0.28, 0.577, 0.491]
accuracy = [0.569, 0.541, 0.401]

plt.figure(figsize=(6,5))

# Scatter plot
plt.scatter(llm_usage, accuracy)

# Annotate points
for i in range(len(datasets)):
    plt.text(llm_usage[i], accuracy[i], datasets[i])

# Axis labels
plt.xlabel("LLM Usage (%)")
plt.ylabel("Exact Match")

# Title
plt.title("Routing Behavior Across Datasets")

# IMPORTANT: Fix axis scale for consistency
plt.xlim(0, 1)
plt.ylim(0, 1)

# Grid
plt.grid(alpha=0.3)

# Save
plt.savefig("../plots/global/routing_behavior.png", dpi=300, bbox_inches='tight')
plt.close()