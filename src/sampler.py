import numpy as np
from typing import Dict

LABEL_TYPES = (
    "$KEEP",
    "$DELETE",
    "$APPEND",
    "$REPLACE",
    "$MERGE",
    "$TRANSFORM_SPLIT",
    "$TRANSFORM_CASE",
    "$TRANSFORM_VERB",
    "$TRANSFORM_AGREEMENT",
)


class TopCategorySampler:
    """
    Category based action sampler

    for each token:
        tokens_actions = actions with the highest value from each category
        randomly select one action from token_actions
    """
    def __init__(self, labels: np.char.array):
        self.encoded_labels_dict = self.categorize(labels)

    @staticmethod
    def categorize(labels: np.char.array) -> Dict[int, np.ndarray]:
        """
        Generate label mask for each category
        """
        categorized_labels_dict = {}
        for i, cat in enumerate(LABEL_TYPES):
            categorized_labels_dict[i] = labels.startswith(cat)
        return categorized_labels_dict

    def sample(self, values: np.ndarray) -> np.ndarray:
        """
        Generate action based on given values
        """
        num_tokens, num_labels = values.shape
        action_indexes = np.arange(num_labels)
        actions = np.zeros(num_tokens, dtype="uint32")
        for tok_i in range(num_tokens):
            tok_actions = []                                # Possible actions for current token
            for cat_mask in self.encoded_labels_dict.values():
                indexes = action_indexes[cat_mask]          # Action indexes of labels in current category
                selected_values = values[tok_i, indexes]    # Values of selected actions
                max_i = np.argmax(selected_values)
                action_index = indexes[max_i]               # Action index with maximum value
                tok_actions.append(action_index)
            actions[tok_i] = np.random.choice(tok_actions)  # Randomly sample action from current token's actions
        return actions


class IndexSampler:
    """
    Generates repeated indexes at given intervals
    """
    def __init__(self, indexes: np.ndarray, interval: int, repeat: int = 2):
        self.indexes = indexes
        self.interval = interval
        self.repeat = repeat
        self.iterator = None
        self.reset()

    def generate(self):
        """
        Generator to initialize the indexes
        """
        for i in range(0, len(self.indexes), self.interval):
            current_indexes = self.indexes[i:i + self.interval]
            current_indexes = np.tile(current_indexes, self.repeat)    # Repeat the current indexes for N times
            np.random.shuffle(current_indexes)                         # Shuffle the current indexes
            for index in current_indexes:
                yield index

    def reset(self):
        """
        Initialize the index generator
        """
        np.random.shuffle(self.indexes)                                # Shuffle the entire indexes
        self.iterator = self.generate()                                # Initialize the generator

    def sample(self) -> int:
        """
        Obtain next index
        """
        i = next(self.iterator, None)                                  # Obtain the next index
        if i is None:                                                  # Previous index generator ended
            self.reset()                                               # Prepare new index generator
            i = next(self.iterator, None)                              # Obtain the next index from the new generator
        return i
