import os
import json
import unicodedata
import numpy as np
from torch import Tensor
import multiprocessing as mp
from typing import Tuple, Iterable, List, Dict, Any, Union
from nltk.translate.gleu_score import sentence_gleu

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
VERB_VOCAB_PATH = os.path.join(ROOT_PATH, r"data/vocabs/verb-form-vocab.txt")
# Punctuation normalisation dictionary derived from the BEA2019 Shared Task style JSON to M2 conversion script
NORM_DICT = {
    "’": "'",
    "´": "'",
    "‘": "'",
    "′": "'",
    "`": "'",
    '“': '"',
    '”': '"',
    '˝': '"',
    '¨': '"',
    '„': '"',
    '『': '"',
    '』': '"',
    '–': '-',
    '—': '-',
    '―': '-',
    '¬': '-',
    '、': ',',
    '，': ',',
    '：': ':',
    '；': ';',
    '？': '?',
    '！': '!',
    'ِ': '',
    '\u200b': ''
}
NORM_DICT = {ord(k): v for k, v in NORM_DICT.items()}
REPLACE_DICT = {
    '``': '"',
    "''": '"',
}
START_TOKEN = "$START"


def freeze(params: Iterable[Tensor], requires_grad: bool = False):
    """
    Freeze parameters by changing its requires_grad attribute
    """
    for param in params:
        param.requires_grad = requires_grad


def stack_padding(it, dtype="float32"):
    """
    Stack and pad arrays to match the length of the longest array
    Source: https://stackoverflow.com/a/53052599
    """
    def resize(row, size):
        new = np.array(row)
        new.resize(size)
        return new

    row_length = len(max(it, key=len))     # get longest row length
    mat = np.array([resize(row, row_length) for row in it], dtype=dtype)
    return mat


def clean_text(text):
    """
    Cleans the raw text characters
    """
    text = text.translate(NORM_DICT)       # Normalize punctuations
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    for old, new in REPLACE_DICT.items():
        text = text.replace(old, new)
    return text


def load_json(data_path: str) -> Dict[Any, Any]:
    """
    Load JSON file
    """
    with open(data_path, "r") as fp:
        return json.load(fp)


def write_json(data_path: str, data: Union[dict, Iterable[dict]], indent: int = 4, **kwargs):
    """
    Write to a JSON file
    """
    with open(data_path, "w") as fp:
        json.dump(data, fp, indent=indent, **kwargs)


def load_text(data_path: str) -> List[str]:
    """
    Load text data
    """
    with open(data_path, "r") as fp:
        data = [line.strip() for line in fp if line.strip()]
    return data


def get_verb_form_dicts():
    """
    Obtain verb transformation dictionaries
    Source: https://github.com/grammarly/gector/blob/daf42a1b48574523b77fd25b05ededefc79d5b2e/utils/helpers.py#L20
    """
    encoded_verbs, decoded_verbs = {}, {}
    with open(VERB_VOCAB_PATH, "r", encoding="utf-8") as fp:
        for line in fp:
            words, tags = line.split(":")
            word1, word2 = words.split("_")
            tag1, tag2 = tags.split("_")
            decode_key = f"{word1}_{tag1}_{tag2.strip()}"
            if decode_key not in decoded_verbs:
                encoded_verbs[words] = tags
                decoded_verbs[decode_key] = word2
    return encoded_verbs, decoded_verbs


ENCODE_VERB_DICT, DECODE_VERB_DICT = get_verb_form_dicts()


def apply_transform_case(tok_text: str, label: str) -> Tuple[str, bool]:
    """
    Apply case transformation to the token text
    """
    if label == "$TRANSFORM_CASE_LOWER":
        new_text = tok_text.lower()
    elif label == "$TRANSFORM_CASE_UPPER":
        new_text = tok_text.upper()
    elif label == "$TRANSFORM_CASE_CAPITAL":
        new_text = tok_text.capitalize()
    elif label == "$TRANSFORM_CASE_CAPITAL_1":
        if len(tok_text) <= 1:
            new_text = tok_text
        else:
            new_text = f"{tok_text[0]}{tok_text[1:].capitalize()}"
    elif label == "$TRANSFORM_CASE_UPPER_-1":
        if len(tok_text) <= 1:
            new_text = tok_text
        else:
            new_text = f"{tok_text[:-1].upper()}{tok_text[-1]}"
    else:
        raise ValueError(f"Invalid '$TRANSFORM_CASE' label. Got '{label}'!")
    return new_text, (new_text == tok_text)


def apply_transform_plural(tok_text: str, label: str) -> Tuple[str, bool]:
    """
    Apply agreement transformation to the token text
    """
    if not tok_text:
        return tok_text, True
    if label == "$TRANSFORM_AGREEMENT_SINGULAR":
        if tok_text[-1] != "s":
            return tok_text, True
        else:
            return tok_text[:-1], False
    elif label == "$TRANSFORM_AGREEMENT_PLURAL":
        if tok_text[-1] == "s":
            return tok_text, True
        else:
            return f"{tok_text}s", False
    else:
        raise ValueError(f"Invalid '$TRANSFORM_AGREEMENT' label. Got '{label}'!")


def apply_transform_verb(tok_text: str, label: str) -> Tuple[str, bool]:
    """
    Apply verb transformation to the token text
    """
    verb_transform = label.split("_", 2)[-1]              # Extract "VERB1_VERB2" from "$TRANSFORM_VERB_VERB1_VERB2"
    encoded_req = f"{tok_text}_{verb_transform}"
    target_verb = DECODE_VERB_DICT.get(encoded_req)
    if target_verb is None:
        target_verb = tok_text
        verb_not_found = True
    else:
        verb_not_found = False
    return target_verb, verb_not_found


def decode(tokens: List[str], labels: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Generate new tokens by applying the labels to current tokens
    """
    # Make a copy to prevent changing the original lists
    tokens = tokens.copy()
    labels = labels.copy()
    num_tokens = len(tokens)
    num_labels = len(labels)
    assert num_tokens == num_labels, f"Number of tokens and labels must be same. Got {num_tokens} and {num_labels}!"
    invalid_label_masks = np.zeros(num_labels, dtype="uint32")
    for token_i in reversed(range(num_tokens)):   # Appends and deletes don't affect the next edits in reversed order
        tok_text = tokens[token_i]
        label = labels[token_i]
        if label == "$KEEP":
            continue
        elif label == "$DELETE":
            del tokens[token_i]
        elif label.startswith("$APPEND_"):
            append_index = token_i + 1
            _, append_word = label.split("_", 1)
            tokens.insert(append_index, append_word)
        elif label.startswith("$REPLACE_"):
            replace_word = label.split("_", 1)[1]
            tokens[token_i] = replace_word
            invalid_label_masks[token_i] = (tok_text == replace_word)
        elif label.startswith("$MERGE_"):
            sep = "" if label == "$MERGE_SPACE" else "-"
            tok_span = tokens[token_i:token_i + 2]
            if len(tok_span) > 1:
                merged_text = sep.join(tok_span)
                tokens[token_i] = merged_text
                # Remove the 2nd token as it is already merged to the 1st token
                del tokens[token_i + 1]
            else:
                invalid_label_masks[token_i] = True
        elif label.startswith("$TRANSFORM_SPLIT_HYPHEN"):
            split_texts = tok_text.split("-")
            num_split_texts = len(split_texts)
            if num_split_texts > 0:
                tokens[token_i] = split_texts[0]               # Update original token with the first split token
                for i in range(1, num_split_texts):
                    tokens.insert(token_i + i, split_texts[i])  # Add remaining tokens
            invalid_label_masks[token_i] = num_split_texts < 2
        elif label.startswith("$TRANSFORM_AGREEMENT_"):
            tokens[token_i], invalid_label_masks[token_i] = apply_transform_plural(tok_text, label)
        elif label.startswith("$TRANSFORM_CASE_"):
            tokens[token_i], invalid_label_masks[token_i] = apply_transform_case(tok_text, label)
        elif label.startswith("$TRANSFORM_VERB_"):
            tokens[token_i], invalid_label_masks[token_i] = apply_transform_verb(tok_text, label)
        else:
            raise NotImplementedError(f"Cannot handle this label: {label}")
    if len(tokens) == 0:                                        # Handle situation where all tokens are deleted
        tokens = [""]
    return tokens, invalid_label_masks


def apply_labels(labels: List[str], positions: List[int], tokens: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Modified version of "decode()" to apply labels to tokens based on the given token indexes
    """
    tokens = tokens.copy()
    num_tokens = len(tokens)
    assert all(pos < num_tokens for pos in positions)
    assert len(labels) == len(positions)
    invalid_label_masks = np.zeros(num_tokens, dtype="uint32")
    for label, token_i in zip(labels, positions):
        tok_text = tokens[token_i]
        if label == "$KEEP":
            continue
        elif label == "$DELETE":
            del tokens[token_i]
        elif label.startswith("$APPEND_"):
            append_index = token_i + 1
            _, append_word = label.split("_", 1)
            tokens.insert(append_index, append_word)
        elif label.startswith("$REPLACE_"):
            replace_word = label.split("_", 1)[1]
            tokens[token_i] = replace_word
            invalid_label_masks[token_i] = (tok_text == replace_word)
        elif label.startswith("$MERGE_"):
            sep = "" if label == "$MERGE_SPACE" else "-"
            tok_span = tokens[token_i:token_i + 2]
            if len(tok_span) > 1:
                merged_text = sep.join(tok_span)
                tokens[token_i] = merged_text
                del tokens[token_i + 1]                 # Remove the 2nd token as it is already merged to the 1st token
            else:
                invalid_label_masks[token_i] = True
        elif label.startswith("$TRANSFORM_SPLIT_HYPHEN"):
            split_texts = tok_text.split("-")
            num_split_texts = len(split_texts)
            if num_split_texts > 0:
                tokens[token_i] = split_texts[0]               # Update original token with the first split token
                for i in range(1, num_split_texts):
                    tokens.insert(token_i + i, split_texts[i])  # Add remaining tokens
            invalid_label_masks[token_i] = num_split_texts < 2
        elif label.startswith("$TRANSFORM_AGREEMENT_"):
            tokens[token_i], invalid_label_masks[token_i] = apply_transform_plural(tok_text, label)
        elif label.startswith("$TRANSFORM_CASE_"):
            tokens[token_i], invalid_label_masks[token_i] = apply_transform_case(tok_text, label)
        elif label.startswith("$TRANSFORM_VERB_"):
            tokens[token_i], invalid_label_masks[token_i] = apply_transform_verb(tok_text, label)
        else:
            raise NotImplementedError(f"Cannot handle this label: {label}")
    return tokens, invalid_label_masks


def greedy_search(state: List[str], references: List[List[str]], labels: List[str], num_processes: int = None) -> np.ndarray:
    """
    Greedy Search algorithm to apply all labels to each token and select labels
    with the highest score
    """
    num_tokens = len(state)
    actions = np.zeros(num_tokens, dtype="int32")
    with mp.Pool(processes=num_processes) as pool:
        current_state = state
        for tok_i in reversed(range(num_tokens)):
            # Generate new states by applying all labels
            iterable = (([label], [tok_i], current_state) for label in labels)
            outputs = pool.starmap(apply_labels, iterable)
            new_states, _ = zip(*outputs)
            # Calculate GLEU score for all new states
            iterable = ((references, tokens) for tokens in new_states)
            rewards = pool.starmap(sentence_gleu, iterable)
            # Get action with the highest reward
            best_action = np.argmax(rewards)
            actions[tok_i] = best_action
            current_state = new_states[best_action]
    return actions
