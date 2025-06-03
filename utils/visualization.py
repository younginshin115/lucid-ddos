# utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np

def plot_label_distribution(y, label_map=None, title="Label Distribution", save_path="label_dist.png"):
    labels, counts = np.unique(y, return_counts=True)
    if label_map:
        class_names = [label_map.get(int(l), f"Class {l}") for l in labels]
    else:
        class_names = [f"Class {l}" for l in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(class_names, counts)
    plt.xlabel("Class")
    plt.ylabel("Sample Count")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()