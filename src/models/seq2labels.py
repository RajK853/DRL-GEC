import torch
from torch import nn
from typing import Dict, List, Tuple, Any
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from src.models.utils import reduce_mean
from src.utils import stack_padding, START_TOKEN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cache_dir = r"~/.cache/huggingface"


class PretrainedEncoder(nn.Module):
    def __init__(self, model_name: str, tokenizer_config: Dict[str, Any] = None, transformer_config: Dict[str, Any] = None):
        super(PretrainedEncoder, self).__init__()
        tokenizer_config = tokenizer_config or {"use_fast": True}
        transformer_config = transformer_config or {"output_attentions": False}
        # Init tokenizer and transformer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_config, cache_dir=cache_dir)
        self.config = AutoConfig.from_pretrained(model_name, **transformer_config, cache_dir=cache_dir)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config, cache_dir=cache_dir)
        # Remove the last pooler layer as it is not present in AutoModelForTokenClassification
        self.transformer.pooler = None
        self.return_offsets_mapping = isinstance(self.tokenizer, PreTrainedTokenizerFast)
        # self.add_token(START_TOKEN)

    def add_token(self, token: str):
        self.tokenizer.add_tokens([token])
        self.tokenizer.vocab[token] = len(self.tokenizer) - 1

    def forward(self, tokens: List[str]) -> torch.Tensor:
        token_data = self.tokenize_batch(tokens)
        aligns = self.get_alignments(token_data["offset_mapping"], token_data["lengths"])
        trf_out = self.transformer(
                input_ids=token_data["input_ids"].to(device),
                attention_mask=token_data["attention_mask"].to(device),
        )
        encoded_out = reduce_mean(trf_out.last_hidden_state, aligns).to(device)
        return encoded_out

    def tokenize_batch(self, tokens: List[str]) -> Dict[str, Any]:
        token_data = self.tokenizer(
                [" ".join(toks) for toks in tokens],
                add_special_tokens=False,
                return_attention_mask=True,
                return_offsets_mapping=self.return_offsets_mapping,
                return_tensors="pt",
                return_token_type_ids=None,
                padding="longest",
        )
        token_data["lengths"] = [[len(tok) for tok in toks] for toks in tokens]
        return token_data

    @staticmethod
    def get_alignments(offsets: torch.Tensor, token_lens: List[List[int]]) -> torch.Tensor:
        aligns = []
        batch_size, pad_size = offsets.shape[:2]
        offset_lens = offsets[:, :, 1] - offsets[:, :, 0]      # Obtain offset lengths by subtracting start-end indexes
        for i in range(batch_size):
            seq_token_lens = token_lens[i]
            seq_offset_lens = offset_lens[i]
            start_index = 0
            seq_aligns = []
            for tok_i, tok_len in enumerate(seq_token_lens):
                cum_offset = 0                                 # Cumulative offset
                for offset_index in range(start_index, pad_size):
                    cum_offset += seq_offset_lens[offset_index]
                    if cum_offset == tok_len:                  # Cumulative offset matches current token length
                        seq_aligns.append(offset_index - start_index + 1)
                        start_index = offset_index + 1         # Next starting index
                        break
            aligns.append(seq_aligns)
        padded_aligns = stack_padding(aligns, dtype="int32")   # Pad the aligns as int32
        return torch.from_numpy(padded_aligns)


class Seq2Labels(nn.Module):
    def __init__(self, encoder_model: PretrainedEncoder, num_labels: int, dropout: float = 0.1):
        super(Seq2Labels, self).__init__()
        self.encoder_model = encoder_model.to(device)
        encoder_output_size = encoder_model.config.hidden_size
        self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(encoder_output_size, num_labels),
        ).to(device)

    def forward(self, tokens: List[str]) -> torch.Tensor:
        encoded_out = self.encoder_model(tokens)
        logit_out = self.classifier(encoded_out)
        return logit_out


class Seq2LabelsV2(nn.Module):
    def __init__(self, encoder_model: PretrainedEncoder, num_categories: int, num_labels: int, dropout: float = 0.1):
        super(Seq2LabelsV2, self).__init__()
        self.encoder_model = encoder_model.to(device)
        encoder_output_size = encoder_model.config.hidden_size
        self.category_classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(encoder_output_size, num_categories),
                nn.Softmax(dim=-1),
        ).to(device)
        self.label_classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(encoder_output_size, num_labels),
        ).to(device)

    def forward(self, tokens: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_out = self.encoder_model(tokens)
        labels_logit_out = self.label_classifier(encoded_out)
        categories_logit_out = self.category_classifier(encoded_out)
        return labels_logit_out, categories_logit_out
