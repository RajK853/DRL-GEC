import numpy as np
from typing import List
from functools import lru_cache
from cdifflib import CSequenceMatcher as SequenceMatcher


Tokens = List[str]
Actions = np.ndarray
Labels = np.char.array
# Map edit type to label types
EDIT2LABELS = {
    "equal": ["$KEEP"],
    "delete": ["$DELETE"],
    "insert": ["$APPEND"],
    "replace": ["$REPLACE", "$MERGE", "$TRANSFORM", "$UNKNOWN"],
}


class EditMaskGenerator:
    """
    Class to generate edit types based on action indices and labels
    """

    def __init__(self, labels: Labels):
        self.labels = labels
        self.encoded_labels = self.encode(labels)

    @staticmethod
    def encode(labels: Labels) -> Labels:
        encoded_labels = np.char.array(["equal"] * len(labels), itemsize=7)
        for edit_type, label_types in EDIT2LABELS.items():
            for label_type in label_types:
                encoded_labels[labels.startswith(label_type)] = edit_type
        return encoded_labels

    @staticmethod
    def get_edit_mask(tokens: Tokens, ref_tokens_list: List[Tokens]) -> Labels:
        """
        Generate SequenceMatcher edits for given token and references
        """
        # Choose reference which is most similar to the tokens
        if len(ref_tokens_list) == 1:
            ref_tokens = ref_tokens_list[0]
        else:
            sim_scores = [SequenceMatcher(None, tokens, ref_tokens).ratio() for ref_tokens in ref_tokens_list]
            ref_tokens = ref_tokens_list[np.argmax(sim_scores)]
        # Generate edit mask
        edits = SequenceMatcher(None, tokens, ref_tokens).get_opcodes()
        edit_mask = np.char.array(["equal"] * len(tokens), itemsize=7)
        for edit in edits:
            edit_type, i, j, _, _ = edit
            if edit_type == "equal":
                continue
            elif edit_type == "insert" and i == j:
                i -= 1                                # Adjust i such that tokens[i:j] = token to use the insert edit
            elif edit_type == "replace" and j-i > 1:  # Change more than 1 consecutive replace edits to mixed edit
                edit_type = "mixed"
            edit_mask[i:j] = edit_type
        return edit_mask

    def actions_to_edits(self, actions: Actions) -> Labels:
        """
        Convert action indices (int) into edit types (str)
        """
        return self.encoded_labels[actions]

    def actions_to_labels(self, actions: Actions) -> Labels:
        """
        Convert action indices (int) into action labels (str)
        """
        return self.labels[actions]

    @lru_cache()
    def label_to_action(self, label: str) -> int:
        """
        Convert a action label into action index
        """
        index, *_ = np.where(self.labels == label)
        return index.item()

    def labels_to_actions(self, labels: Labels) -> Actions:
        """
        Convert action labels into action indices
        """
        return np.vectorize(self.label_to_action)(labels)

    @lru_cache()
    def edit_to_actions(self, edit: str) -> Actions:
        """
        Get the action indices of the action edit type
        """
        if edit == "mixed":
            actions = [np.where(self.encoded_labels == e)[0] for e in ("delete", "insert", "replace")]
            actions = np.concatenate(actions)
        else:
            actions, *_ = np.where(self.encoded_labels == edit)
        return actions

    @lru_cache()
    def edit_to_labels(self, edit: str) -> Labels:
        """
        Get the action labels of the action edit type
        """
        actions = self.edit_to_actions(edit)
        labels = self.actions_to_labels(actions)
        return labels


class IndexSampler:
    """
    Class to generate repeated indexes at given intervals
    """
    def __init__(self, indexes: np.ndarray, interval: int, repeat: int = 2, consecutive: bool = True):
        self.indexes = indexes
        self.interval = interval
        self.repeat = repeat
        self.consecutive = consecutive
        self.iterator = None
        self.reset()

    def generate(self):
        """
        Generator to initialize the indexes
        """
        for i in range(0, len(self.indexes), self.interval):
            current_indexes = self.indexes[i:i + self.interval]
            if self.repeat > 1:
                if self.consecutive:
                    for index in current_indexes:
                        for _ in range(self.repeat):
                            yield index
                else:
                    current_indexes = np.tile(current_indexes, self.repeat)  # Repeat the current indexes for N times
                    np.random.shuffle(current_indexes)                       # Shuffle the current indexes
                    for index in current_indexes:
                        yield index
            else:
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
