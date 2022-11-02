import os
import re
import json
import torch
import socket
import unicodedata
import numpy as np
from torch import Tensor
from tqdm.auto import tqdm
from itertools import islice
import multiprocessing as mp
from typing import Iterable, List, Union
from rapidfuzz.distance import Levenshtein
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
    re.compile('``'): '"',
    re.compile("''"): '"',
}
REF_REPLACE_DICT = {
    re.compile(r"(\B)('m)(\b)"): r"\1am\3",      # 'm -> am
    re.compile(r"(\B)(n't)(\b)"): r" not\3",     # haven't -> have not
    re.compile(r"(\b)(n't)(\b)"): r"\1not\3",    # have n't -> have not
    re.compile(r"(\b)(i)(\b)"): r"\1I\3",        # i -> I
    re.compile(r"(\W)(' s)(\b)"): r"\1's\3",     # ' s -> 's
}
START_TOKEN = "$START"
UNK_TOKEN = "$UNKNOWN"
LABELS_SEP = "SEPL__SEPR"
TOK_LABEL_SEP = "SEPL|||SEPR"


def is_gce_instance():
    """
    Check if it's GCE instance via DNS lookup to metadata server.
    Source:
    https://stackoverflow.com/a/58619342
    """
    try:
        socket.getaddrinfo('metadata.google.internal', 80)
    except socket.gaierror:
        return False
    return True


def freeze_params(model, requires_grad=False, num_layers=0, optim=None, lr=None):
    freeze(model.parameters(), requires_grad=requires_grad, num_layers=num_layers)
    param_status = tuple(not param.requires_grad for param in model.parameters())
    print(f"Number of frozen parameters: {sum(param_status)}/{len(param_status)}")
    if optim is not None:
        assert lr is not None, "Learning rate not given!"
        for param_group in optim.param_groups:
            param_group['lr'] = lr


def discount_cumsum(data, discount):
    cumsum = 0.0
    num_data = len(data)
    discounted_values = np.zeros(num_data, dtype="float32")
    for i in reversed(range(num_data)):
        cumsum = discounted_values[i] = data[i] + discount*cumsum
    return discounted_values


def scale(values):
    values = np.array(values)
    values = (values - values.mean())/(values.std() + 1e-8)
    return list(values)


def adaptive_sentence_gleu(
        references: List[List[str]],
        tokens: List[str],
        min_len: int = 1,      # Minimum N-gram length
        max_len: int = 4,      # Maximum N-gram length
        adapt: bool = True,    # When true, overrides the max_len based on reference length
        div_len: int = 4,      # Divisor to calculate max_len in adaptive mode
        clip_min: int = 2,     # Minimum max_len in adaptive mode
        clip_max: int = 20,    # Maximum max_len in adaptive mode
) -> float:
    """
    Calculates Adaptive-GLEU that adjusts the `max_len` parameter for GLEU score based on the reference length
    """
    if adapt:
        max_len = max(len(ref) for ref in references)//div_len
        max_len = np.clip(max_len, clip_min, clip_max)
    ref_gleu = sentence_gleu(references, tokens, min_len=min_len, max_len=max_len)
    return ref_gleu


def lev_dist_reward(dist: int, ref_len: int) -> float:
    """
    Calculates score based on Levenshtein distance
    """
    return np.exp(-(dist ** 2) / (2 * ref_len))


def get_lev_dist(tokens_a, tokens_b):
    return Levenshtein.distance(tokens_a, tokens_b, weights=(1, 1, 2))


def limit_corrections(actions, max_num, default_action=0):
    correction_mask = actions != default_action
    if sum(correction_mask) < max_num:
        return actions
    action_indexes, *_ = np.where(correction_mask)
    selected_indexes = np.random.choice(action_indexes, size=max_num, replace=False)
    new_actions = np.full(actions.shape, fill_value=default_action)
    new_actions[selected_indexes] = actions[selected_indexes]
    return new_actions


def filter_by_order(labels, label_types=None):
    def has_label_type(l_type):
        return any(label.startswith(l_type) for label in labels)

    def filter_labels(l_type):
        return [label if label.startswith(l_type) else "$KEEP" for label in labels]

    if label_types is None:
        label_types = ["$APPEND", "$MERGE", "$TRANSFORM", "$REPLACE", "DELETE"]
    for label_type in label_types:
        if has_label_type(label_type):
            return filter_labels(label_type)
    return labels


def add_start(tokens):
    if tokens and tokens[0] != START_TOKEN:
        tokens.insert(0, START_TOKEN)
    return tokens


def remove_start(tokens):
    if tokens and tokens[0] == START_TOKEN:
        del tokens[0]
    return tokens


@torch.no_grad()
def predict(policy, label_vocab, token_list, filter_labels=False):
    logits = policy(token_list)
    action_list = logits.argmax(-1).cpu().numpy()
    label_list = label_vocab[action_list]
    if filter_labels:
        label_list = [filter_by_order(labels) for labels in label_list]
    new_tokens = [apply_labels(tokens, labels[:len(tokens)]) for tokens, labels in zip(token_list, label_list)]
    return new_tokens


def iterative_prediction(policy, label_vocab, texts, num_iter=5, filter_labels=False, insert_start=True, batch_size=16, verbose=True):
    sent_list = [sent.split() for sent in texts]
    orig_sent_lengths = [len(sent) for sent in sent_list]
    if insert_start:                                              # Add start token
        sent_list = [add_start(tokens) for tokens in sent_list]
    sent_indexes = list(range(len(sent_list)))                    # Indexes of sentence to be predicted
    for iter_num in range(1, num_iter + 1):
        next_sent_indexes = []                                    # Store indexes of docs for the next iteration
        iterator = tqdm(sent_indexes, total=len(sent_indexes), desc=f"Iteration {iter_num}") if verbose else sent_indexes
        for indexes_batch in minibatch(iterator, batch_size):
            tokens_batch = [sent_list[sent_i] for sent_i in indexes_batch]
            new_tokens_batch = predict(policy, label_vocab, tokens_batch, filter_labels=filter_labels)
            for sent_i, tokens, new_tokens in zip(indexes_batch, tokens_batch, new_tokens_batch):
                sent_list[sent_i] = new_tokens
                # Check that the sentence length is between 0 and 1.5x original length
                is_sent_len_valid = 0 < len(new_tokens) < 1.5*orig_sent_lengths[sent_i]
                if new_tokens != tokens and is_sent_len_valid:
                    # Queue the doc for next iteration if it changed and sentence length is in valid range
                    next_sent_indexes.append(sent_i)
        if not next_sent_indexes:                                 # No sentence left for the next iteration
            break
        sent_indexes = next_sent_indexes
    if insert_start:                                             # Remove start token
        sent_list = [remove_start(tokens) for tokens in sent_list]
    return [" ".join(tokens) for tokens in sent_list]


def remove_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def minibatch(seqs, size):
    seqs = iter(seqs)
    while True:
        batch = list(islice(seqs, size))
        if not batch:
            break
        yield batch


def is_solvable(text, references, labels):
    from gector.utils import preprocess_data

    for reference in references:
        alignment = preprocess_data.align_sequences(text, reference)
        for tok_label in alignment.split():
            token, tok_labels = tok_label.split(TOK_LABEL_SEP)
            if any(label not in labels for label in tok_labels.split(LABELS_SEP)):
                return False
    return True


def filter_by_num_ref(data, min_refs=2):
    if min_refs == 1:
        return data
    filtered_data = [data_dict for data_dict in data if len(data_dict["references"]) >= min_refs]
    return filtered_data


def filter_solvable(data_dict, label_vocab):
    if is_solvable(data_dict["text"], data_dict["references"], label_vocab):
        return data_dict
    return None


def filter_by_solvable(data_list, labels, batch_size=64, num_processes=None):
    with mp.Pool(processes=num_processes) as pool:
        iterator = minibatch(((data_dict, labels) for data_dict in data_list), batch_size)
        filtered_data = []
        pbar = tqdm(desc="Filtering Unsolvable Sentences", total=len(data_list), unit_scale=True)
        for chunk in iterator:
            results = pool.starmap(filter_solvable, chunk)
            for i, data_dict in enumerate(results, start=1):
                if data_dict is not None:
                    filtered_data.append(data_dict)
            pbar.update(i)
            pbar.refresh()
        pbar.close()
    return filtered_data


def precision(tp: int, fp: int) -> float:
    """
    Calculate precision
    """
    if tp == 0:
        return 0.0
    return tp/(tp+fp)


def recall(tp: int, fn: int) -> float:
    """
    Calculate recall
    """
    if tp == 0:
        return 0.0
    return tp/(tp+fn)


def f_score(p: float, r: float, beta: float = 1.0) -> float:
    """
    Calculate F-beta score
    """
    if p*r == 0.0:
        return 0.0
    beta_2 = beta**2
    return ((1+beta_2) * p * r) / (beta_2 * p + r)


def filter_correct(train_data: List[dict], correct_percent: float) -> List[dict]:
    """
    Filter the dataset to retain given percent of correct sentences
    """
    correct_indexes = []
    incorrect_indexes = []
    for i, example in enumerate(train_data):
        if example["text"] in example["references"]:
            correct_indexes.append(i)
        else:
            incorrect_indexes.append(i)
    filter_num = round(correct_percent*len(correct_indexes))
    filtered_indexes = list(np.random.choice(correct_indexes, size=filter_num))
    filtered_indexes += incorrect_indexes
    filtered_data = [train_data[i] for i in sorted(filtered_indexes)]
    return filtered_data


def freeze(params: Iterable[Tensor], requires_grad: bool = False, num_layers: int = 0):
    """
    Freeze parameters by changing its requires_grad attribute
    """
    params = tuple(params)
    if num_layers == 0:
        num_layers = len(params)
    elif num_layers < 0:
        num_layers = len(params) + num_layers
        requires_grad = not requires_grad

    for i, param in enumerate(params):
        if i == num_layers:
            requires_grad = not requires_grad
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


def clean_text(text, is_ref=False):
    """
    Cleans the raw text characters
    """
    text = text.translate(NORM_DICT)                   # Normalize punctuations
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    for pattern, sub in REPLACE_DICT.items():          # Normalize some characters/words
        text = pattern.sub(sub, text)
    if is_ref:                                         # Normalize some characters/words for references
        for pattern, sub in REF_REPLACE_DICT.items():
            text = pattern.sub(sub, text)
    return text


def load_json(data_path: str) -> Union[dict, List[dict]]:
    """
    Load JSON file
    """
    with open(data_path, "r") as fp:
        return json.load(fp)


def write_json(data_path: str, data: Union[dict, List[dict]], indent: int = 4, **kwargs):
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


def write_text(data_list: List[str], data_path: str):
    with open(data_path, "w") as fp:
        for line in data_list:
            fp.write(f"{line}\n")


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


def apply_transform_case(tok_text: str, label: str) -> str:
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
    return new_text


def apply_transform_plural(tok_text: str, label: str) -> str:
    """
    Apply agreement transformation to the token text
    """
    if not tok_text:
        return tok_text
    if label == "$TRANSFORM_AGREEMENT_SINGULAR":
        if tok_text[-1] != "s":
            return tok_text
        else:
            return tok_text[:-1]
    elif label == "$TRANSFORM_AGREEMENT_PLURAL":
        if tok_text[-1] == "s":
            return tok_text
        else:
            return f"{tok_text}s"
    else:
        raise ValueError(f"Invalid '$TRANSFORM_AGREEMENT' label. Got '{label}'!")


def apply_transform_verb(tok_text: str, label: str) -> str:
    """
    Apply verb transformation to the token text
    """
    verb_transform = label.split("_", 2)[-1]              # Extract "VERB1_VERB2" from "$TRANSFORM_VERB_VERB1_VERB2"
    encoded_req = f"{tok_text}_{verb_transform}"
    target_verb = DECODE_VERB_DICT.get(encoded_req)
    if target_verb is None:
        target_verb = tok_text
    return target_verb


def apply_labels(tokens: List[str], labels: List[str]) -> List[str]:
    """
    Generate new tokens by applying the labels to current tokens
    """
    positions = list(range(len(tokens)))
    return apply_labels_at(tokens, labels, positions=positions)


def apply_labels_at(tokens: List[str], labels: List[str], positions: List[int]) -> List[str]:
    """
    Modified version of "apply_labels" to apply labels to tokens based on the given token indexes
    """
    tokens = [*tokens]
    num_tokens = len(tokens)
    assert all(pos < num_tokens for pos in positions)
    assert len(labels) == len(positions)
    if len(labels) > 1:
        # Sort labels in descending order of the positions
        labels, positions = zip(*sorted(zip(labels, positions), key=lambda x: x[1], reverse=True))
    for label, token_i in zip(labels, positions):
        tok_text = tokens[token_i]
        if label in ("$KEEP", UNK_TOKEN):
            continue
        elif label == "$DELETE":
            del tokens[token_i]
        elif label.startswith("$APPEND_"):
            _, append_word = label.split("_", 1)
            tokens.insert(token_i + 1, append_word)
        elif label.startswith("$REPLACE_"):
            _, replace_word = label.split("_", 1)
            tokens[token_i] = replace_word
        elif label.startswith("$MERGE_"):
            sep = "" if label == "$MERGE_SPACE" else "-"
            tok_span = tokens[token_i:token_i + 2]              # Take the current and next tokens
            if len(tok_span) == 2:
                merged_text = sep.join(tok_span)
                tokens[token_i] = merged_text
                del tokens[token_i + 1]                         # Remove the 2nd token as it is already merged
        elif label.startswith("$TRANSFORM_SPLIT_HYPHEN"):
            split_texts = tok_text.split("-")
            num_split_texts = len(split_texts)
            if num_split_texts > 0:
                tokens[token_i] = split_texts[0]                # Update original token with the first split token
                for i in range(1, num_split_texts):
                    tokens.insert(token_i + i, split_texts[i])  # Add remaining tokens
        elif label.startswith("$TRANSFORM_AGREEMENT_"):
            tokens[token_i] = apply_transform_plural(tok_text, label)
        elif label.startswith("$TRANSFORM_CASE_"):
            tokens[token_i] = apply_transform_case(tok_text, label)
        elif label.startswith("$TRANSFORM_VERB_"):
            tokens[token_i] = apply_transform_verb(tok_text, label)
        else:
            raise NotImplementedError(f"Cannot handle this label: {label}")
    if len(tokens) == 0:                                        # Handle situation where all tokens are deleted
        tokens = [" "]
    return tokens
