from tqdm.auto import tqdm
from collections import defaultdict

from src.utils import stack_padding, TOK_LABEL_SEP, LABELS_SEP, UNK_TOKEN


def process_sent(sent, label_vocab):
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
    batch = defaultdict(list)
    for data_dict in data_batch:
        for k, v in data_dict.items():
            batch[k].append(v)
        batch["masks"].append([1]*len(data_dict["labels"]))
    batch["labels"] = stack_padding(batch["labels"], dtype="int64")
    batch["masks"] = stack_padding(batch["masks"], dtype="bool")
    batch["labels"][~batch["masks"]] = -100                              # Set ignore index
    return batch
