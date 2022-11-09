import torch
from torch import nn
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from src import utils
from src.models.pooling import reduce_mean


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_TOKENIZER_CFG = {"use_fast": True}
DEFAULT_TRANSFORMER_CONFIG = {"output_attentions": False}


class PretrainedEncoder(nn.Module):
    def __init__(
            self,
            model_name: str,
            tokenizer_config: Dict[str, Any] = None,
            transformer_config: Dict[str, Any] = None,
            add_pooling_layer: bool = False,
    ):
        super(PretrainedEncoder, self).__init__()
        # Init tokenizer and transformer
        tokenizer_config = tokenizer_config or DEFAULT_TOKENIZER_CFG.copy()
        transformer_config = transformer_config or DEFAULT_TRANSFORMER_CONFIG.copy()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_config)
        self.config = AutoConfig.from_pretrained(model_name, **transformer_config)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config, add_pooling_layer=add_pooling_layer)
        self.return_offsets_mapping = isinstance(self.tokenizer, PreTrainedTokenizerFast)
        self.add_tokens([utils.START_TOKEN])

    def add_tokens(self, tokens: List[str]):
        """
        Add tokens to the tokenizer and resize the transformer embeddings
        """
        token_vocab = self.tokenizer.get_vocab().keys()
        new_tokens = [token for token in tokens if token not in token_vocab]
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            self.transformer.resize_token_embeddings(len(self.tokenizer))

    def tokenize_batch(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Tokenize batch of strings into
        """
        token_data = self.tokenizer(
                [" ".join(toks) for toks in tokens],
                add_special_tokens=False,
                return_attention_mask=True,
                return_offsets_mapping=self.return_offsets_mapping,
                return_tensors="pt",
                return_token_type_ids=None,
                padding="longest",
        ).to(device)
        token_data["lengths"] = [[len(tok) for tok in toks] for toks in tokens]
        return token_data

    @staticmethod
    def get_alignments(offsets: torch.Tensor, token_lens: List[List[int]]) -> torch.Tensor:
        """
        Get alignments used to keep track of splitted tokens
        """
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
        padded_aligns = utils.stack_padding(aligns, dtype="int32")   # Pad the aligns as int32
        return torch.from_numpy(padded_aligns)

    def forward(self, tokens: List[str]) -> torch.Tensor:
        """
        Forward function
        """
        token_data = self.tokenize_batch(tokens)
        aligns = self.get_alignments(token_data["offset_mapping"], token_data["lengths"])
        trf_out = self.transformer(
                input_ids=token_data["input_ids"],
                attention_mask=token_data["attention_mask"],
        )
        encoded_out = reduce_mean(trf_out.last_hidden_state, aligns)
        return encoded_out


class Seq2Labels(nn.Module):
    def __init__(self, encoder_model: PretrainedEncoder, num_labels: int, dropout: float = 0.1):
        super(Seq2Labels, self).__init__()
        self.encoder = encoder_model.to(device)
        encoder_output_size = encoder_model.config.hidden_size
        self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(encoder_output_size, num_labels),
        ).to(device)

    def forward(self, tokens: List[str]) -> torch.Tensor:
        encoded_out = self.encoder(tokens)
        logit_out = self.classifier(encoded_out)
        return logit_out


class Seq2LabelsDeeper(nn.Module):
    def __init__(self, encoder_model: PretrainedEncoder, num_labels: int, dropout: float = 0.1):
        super(Seq2LabelsDeeper, self).__init__()
        self.encoder = encoder_model.to(device)
        encoder_output_size = encoder_model.config.hidden_size
        self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(encoder_output_size, encoder_output_size),
                nn.ReLU(),
                nn.Linear(encoder_output_size, num_labels),
        ).to(device)

    def forward(self, tokens: List[str]) -> torch.Tensor:
        encoded_out = self.encoder(tokens)
        logit_out = self.classifier(encoded_out)
        return logit_out

