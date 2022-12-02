from tqdm.auto import tqdm
from collections import defaultdict

from src.utils import stack_padding, TOK_LABEL_SEP, LABELS_SEP, UNK_TOKEN


def process_sent(sent, label_vocab):
    """
    Extract tokens and their labels and handle OOV labels
    """
    tokens = []
    labels = []
    for tok_and_label in sent.split(" "):
        token, raw_labels = tok_and_label.split(TOK_LABEL_SEP)
        label = raw_labels.split(LABELS_SEP)[0]  # Take first label from multiple labels
        if label not in label_vocab:
            label = UNK_TOKEN
        tokens.append(token)
        labels.append(label)
    return tokens, labels


def process_data(data_list, label_vocab, keep_corrects=True):
    """
    Process a list of sentences.
    """
    all_tokens = []
    all_labels = []
    for sent in tqdm(data_list, desc="Processing data", total=len(data_list)):
        tokens, labels = process_sent(sent, label_vocab)
        if not keep_corrects and all(label == "$KEEP" for label in labels):
            continue
        all_tokens.append(tokens)
        all_labels.append(labels)
    print(f"Amount of data after filtering: {len(all_tokens)}")
    return all_tokens, all_labels


def collate_func(data_batch):
    non_pad_masks = []
    batch = defaultdict(list)
    for data_dict in data_batch:
        for k, v in data_dict.items():
            batch[k].append(v)
        non_pad_masks.append([1]*len(data_dict["labels"]))
    # Pad the mask and labels to the longest batch sequence
    non_pad_masks = stack_padding(non_pad_masks, dtype="bool")
    batch["labels"] = stack_padding(batch["labels"], dtype="int64")
    batch["labels"][~non_pad_masks] = -100                         # Set ignore index to padding labels
    return batch
