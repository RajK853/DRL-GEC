import os
import json
import logging
import numpy as np
from typing import Tuple

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


def clean_text(text):
    """
    Cleans the raw text characters
    """
    text = text.translate(NORM_DICT)       # Normalize punctuations
    for old, new in REPLACE_DICT.items():
        text = text.replace(old, new)
    return text


def load_json(data_path):
    with open(data_path, "r") as fp:
        return json.load(fp)


def write_json(data_path, data, indent=4, **kwargs):
    with open(data_path, "w") as fp:
        json.dump(data, fp, indent=indent, **kwargs)


def load_text(data_path):
    with open(data_path, "r") as fp:
        data = [line.strip() for line in fp if line.strip()]
    return data


ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
VERB_VOCAB_PATH = os.path.join(ROOT_PATH, r"data/vocabs/verb-form-vocab.txt")


def get_verb_form_dicts():
    """
    Obtain verb transformation dictionaries
    https://github.com/grammarly/gector/blob/daf42a1b48574523b77fd25b05ededefc79d5b2e/utils/helpers.py#L20
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
    if label == "$TRANSFORM_CASE_LOWER":
        new_text = tok_text.lower()
    elif label == "$TRANSFORM_CASE_UPPER":
        new_text = tok_text.upper()
    elif label == "$TRANSFORM_CASE_CAPITAL":
        new_text = tok_text.capitalize()
    elif label == "$TRANSFORM_CASE_CAPITAL_1":
        new_text = f"{tok_text[0]}{tok_text[1:].capitalize()}"
    elif label == "$TRANSFORM_CASE_UPPER_-1":
        new_text = f"{tok_text[:-1].upper()}{tok_text[-1]}"
    else:
        raise ValueError(f"Invalid '$TRANSFORM_CASE' label. Got '{label}'!")
    return new_text, (new_text == tok_text)


def apply_transform_plural(tok_text: str, label: str) -> Tuple[str, bool]:
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
    verb_transform = label.split("_", 2)[-1]              # Extract "VERB1_VERB2" from "$TRANSFORM_VERB_VERB1_VERB2"
    encoded_req = f"{tok_text}_{verb_transform}"
    target_verb = DECODE_VERB_DICT.get(encoded_req)
    if target_verb is None:
        logging.warning(f"Cannot find verb transformation '{label}' for '{tok_text}'.")
        target_verb = tok_text
        verb_not_found = True
    else:
        verb_not_found = False
    return target_verb, verb_not_found


def decode(tokens, labels):
    # Make a copy to prevent changing the original lists
    tokens = tokens.copy()
    labels = labels.copy()
    num_tokens = len(tokens)
    num_labels = len(labels)
    assert num_tokens == num_labels, f"Number of tokens and labels must be same. Got {num_tokens} and {num_labels}!"
    invalid_label_masks = np.zeros(num_labels, dtype="uint32")
    # Appends and deletes do not affect the next edits if applied from reversed order
    for token_i in reversed(range(num_tokens)):
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
            merged_text = sep.join(tokens[token_i:token_i + 2])
            tokens[token_i] = merged_text
            # Remove the 2nd token as it is already merged to the 1st token
            del tokens[token_i + 1]
        elif label.startswith("$TRANSFORM_SPLIT_HYPHEN"):
            split_texts = tok_text.split("-")
            num_split_texts = len(split_texts)
            tokens[token_i] = split_texts[0]               # Update original token with the first split token
            for i in range(1, num_split_texts):
                tokens.insert(token_i + i, split_texts[i])  # Add remaining tokens
            invalid_label_masks[token_i] = (num_split_texts == 1)
        elif label.startswith("$TRANSFORM_AGREEMENT_"):
            tokens[token_i], invalid_label_masks[token_i] = apply_transform_plural(tok_text, label)
        elif label.startswith("$TRANSFORM_CASE_"):
            tokens[token_i], invalid_label_masks[token_i] = apply_transform_case(tok_text, label)
        elif label.startswith("$TRANSFORM_VERB_"):
            tokens[token_i], invalid_label_masks[token_i] = apply_transform_verb(tok_text, label)
        else:
            raise NotImplementedError(f"Cannot handle this label: {label}")
    return tokens, invalid_label_masks
