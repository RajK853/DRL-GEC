from torch.utils.data import Dataset


class GECDataset(Dataset):
    def __init__(self, tokens, labels, label2index):
        self.tokens = tokens
        self.labels = labels
        self.label2index = label2index

    def __getitem__(self, idx):
        batch = {
            "tokens": self.tokens[idx],
            "labels": [self.label2index[label] for label in self.labels[idx]],
        }
        return batch

    def __len__(self):
        return len(self.labels)
