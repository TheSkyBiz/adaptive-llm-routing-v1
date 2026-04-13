import matplotlib.pyplot as plt
import os


def plot_dataset(name, accuracy, cost):
    output_dir = f"../plots/{name}"
    os.makedirs(output_dir, exist_ok=True)

    labels = ["SLM", "v3", "v1", "v2", "LLM"]

    plt.figure(figsize=(7,5))

    # Filter valid points (skip None)
    valid_points = [(c, a, l) for c, a, l in zip(cost, accuracy, labels) if c is not None]

    costs = [p[0] for p in valid_points]
    accs = [p[1] for p in valid_points]
    lbls = [p[2] for p in valid_points]

    plt.plot(costs, accs, marker='o', linewidth=2)

    # Annotate points
    for c, a, l in valid_points:
        plt.text(c, a, l)

    plt.xlabel("Cost")
    plt.ylabel("Exact Match")
    plt.title(f"{name.upper()} Trade-off")

    plt.grid()
    plt.savefig(f"{output_dir}/tradeoff.png", dpi=300)
    plt.close()


# ----------------------------
# DATA
# ----------------------------

plot_dataset(
    "squad",
    [0.462, 0.545, 0.569, 0.559, 0.588],
    [1.0, 2.94, 3.52, 6.94, 10.0]
)

plot_dataset(
    "squad_v2",
    [0.117, 0.461, 0.504, 0.541, 0.258],
    [1.0, 4.54, 5.30, 6.20, 10.0]
)

# IMPORTANT FIX → use None instead of fake 0
plot_dataset(
    "hotpot_qa",
    [0.248, None, None, 0.401, 0.425],
    [1.0, None, None, 5.42, 10.0]
)