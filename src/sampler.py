import numpy as np
from collections import defaultdict


class WeightedSampler:
    def __init__(self, labels, weight_dict):
        self.labels = labels
        self.weight_dict = weight_dict
        self.num_labels = len(self.labels)
        self.labels_freq = self.gen_label_dist(labels)
        # Initialize label probabilities
        self.label_probs = None
        self.init_weights()

    def init_weights(self, weight_dict=None):
        if weight_dict:
            self.weight_dict = weight_dict
        # Normalize weights based on the number of labels i.e. some category like REPLACE or APPEND has multiple labels
        normalized_weights = {k: v / self.labels_freq[k] for k, v in self.weight_dict.items()}
        weights = np.ones(self.num_labels, dtype="float32")
        for i, label in enumerate(self.labels):
            for label_prefix, weight in normalized_weights.items():
                if label.startswith(label_prefix):
                    weights[i] = weight
                    break
        self.label_probs = weights / sum(weights)           # Ensure sum(label_probs) == 1.0

    @staticmethod
    def gen_label_dist(label_list):
        label_dist = defaultdict(int)
        for label in label_list:
            if label in ("$KEEP", "$DELETE"):
                label_dist[label] += 1
            elif any(label.startswith(k) for k in ("$REPLACE_", "$APPEND_", "$MERGE_")):
                label_type, *_ = label.split("_")
                label_dist[label_type] += 1
            elif label.startswith("$TRANSFORM_"):
                label_type = "_".join(label.split("_")[:2])
                label_dist[label_type] += 1
            else:
                raise NotImplementedError(f"Cannot handle {label} label!")
        return label_dist

    def sample(self, size=1):
        return np.random.choice(self.num_labels, size=size, p=self.label_probs)

    def __call__(self, size=1):
        return self.sample(size)
