import torch
import numpy as np
from typing import List, Dict, Any

from src.sampler import TopCategorySampler
from src.replay_buffer import BatchSample
from src.utils import stack_padding, freeze
from src.envs.gec_env import TOKENS, ACTIONS
from src.models import PretrainedEncoder, Seq2Labels, Seq2LabelsV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDQN:
    def __init__(self, env, lr: float = 1e-3, gamma: float = 0.997, dropout: float = 0.2, grad_norm: float = 5.0):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        # Initialize models
        self.encoder = None
        self.q_model = None
        self.tq_model = None
        # TODO: Remove hard-coding
        model_name = "roberta-base"
        tokenizer_config = {"use_fast": True}
        transformer_config = {"output_attentions": False}
        self.init_models(model_name, tokenizer_config, transformer_config, dropout)
        self.optim = torch.optim.Adam(self.q_model.parameters(), lr=lr)
        # Action sampler
        self.action_sampler = TopCategorySampler(env.labels)
        # Loss functions
        self.mse_criterion = torch.nn.MSELoss(reduction="none")
        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction="none")
        torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), max_norm=grad_norm, norm_type=2.0)
        # Initialize target model
        self.freeze_target(requires_grad=False)
        self.update_target(tau=1.0)

    def init_models(self, model_name: str, tokenizer_config: Dict[str, Any], transformer_config: Dict[str, Any], dropout: float):
        self.encoder = PretrainedEncoder(model_name, tokenizer_config, transformer_config).to(device)
        num_labels = self.env.action_space.n
        kwargs = dict(encoder_model=self.encoder, num_labels=num_labels, dropout=dropout)
        self.q_model = Seq2Labels(**kwargs).to(device)
        self.tq_model = Seq2Labels(**kwargs).to(device)

    def freeze_target(self, requires_grad: bool = False):
        freeze(self.tq_model.classifier.parameters(), requires_grad=requires_grad)

    @torch.no_grad()
    def get_targets(self, batch: BatchSample) -> torch.Tensor:
        next_q = self.tq_model(batch.next_states)
        max_q = torch.zeros(len(batch)).to(device)
        for i, state in enumerate(batch.next_states):
            max_q[i] = next_q[i, :len(state), :].amax(dim=(0, 1), keepdim=False)  # select non-padded section
        terminal_mask = ~torch.tensor(batch.is_terminals, dtype=torch.bool).to(device)
        rewards = torch.tensor(batch.rewards, dtype=torch.float32).to(device)
        return rewards + (terminal_mask * self.gamma * max_q)

    def learn(self, batch: BatchSample) -> torch.Tensor:
        q_values = self.q_model(batch.states).to(device)
        batch_size, seq_size, _ = q_values.shape
        b_indexes = torch.arange(batch_size).unsqueeze(1)
        s_indexes = torch.arange(seq_size)
        padded_actions = stack_padding(batch.actions, dtype="int32")
        q_predictions = q_values[b_indexes, s_indexes, padded_actions]
        # Mask to set padded q_values to zero
        mask = torch.zeros(batch_size, seq_size, dtype=torch.int32).to(device)
        for i, state in enumerate(batch.states):
            num_tokens = len(state)
            mask[i, :num_tokens] = 1
        q_predictions = (mask * q_predictions).sum(-1).div(mask.sum(-1))  # Mean q_prediction per sequence length
        q_targets = self.get_targets(batch)
        loss = self.mse_criterion(q_predictions, q_targets).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss

    def select_action(self, state: TOKENS, eps: float = 0.0) -> ACTIONS:
        with torch.no_grad():
            [q_values] = self.q_model([state]).cpu().numpy()
            if np.random.random() <= eps:
                action = self.action_sampler.sample(q_values)
            else:
                action = q_values.argmax(1)
        assert len(state) == len(action)
        return action

    def update_target(self, tau: float = 1.0):
        src_params = self.q_model.classifier.named_parameters()
        trg_params = self.tq_model.classifier.named_parameters()

        # TODO: Parallelize this
        param_dict = {}
        for ((name, src_value), (_, trg_value)) in zip(src_params, trg_params):
            param_dict[name] = tau * src_value.data + (1 - tau) * trg_value.data
        self.tq_model.classifier.load_state_dict(param_dict)


CATEGORIES = [
    '$KEEP',
    '$DELETE',
    '$MERGE',
    '$APPEND',
    '$REPLACE',
    '$TRANSFORM_CASE',
    '$TRANSFORM_VERB',
    '$TRANSFORM_SPLIT',
    '$TRANSFORM_AGREEMENT'
]


def encode(labels: np.char.array) -> np.ndarray:
    encoded_labels = np.zeros((len(labels), len(CATEGORIES)), dtype="float32")
    for i, label in enumerate(CATEGORIES):
        label_indexes = labels.startswith(label)
        encoded_labels[label_indexes, i] = 1.0
    return encoded_labels


class DDQNv2(DDQN):
    def __init__(self, env, lr: float = 1e-3, gamma: float = 0.997, dropout: float = 0.2, grad_norm: float = 5.0):
        super().__init__(env, lr, gamma, dropout, grad_norm)
        self.encoded_labels = encode(env.labels)

    def init_models(self, model_name: str, tokenizer_config: Dict[str, Any], transformer_config: Dict[str, Any], dropout: float):
        num_labels = self.env.action_space.n
        num_categories = len(CATEGORIES)
        self.encoder = PretrainedEncoder(model_name, tokenizer_config, transformer_config).to(device)
        kwargs = dict(encoder_model=self.encoder, num_categories=num_categories, num_labels=num_labels, dropout=dropout)
        self.q_model = Seq2LabelsV2(**kwargs).to(device)
        self.tq_model = Seq2LabelsV2(**kwargs).to(device)

    def freeze_target(self, requires_grad: bool = False):
        freeze(self.tq_model.labels_classifier.parameters(), requires_grad=requires_grad)
        freeze(self.tq_model.categories_classifier.parameters(), requires_grad=requires_grad)

    @torch.no_grad()
    def get_targets(self, batch: BatchSample) -> torch.Tensor:
        next_q, _ = self.tq_model(batch.next_states)
        max_q = torch.zeros(len(batch)).to(device)
        for i, state in enumerate(batch.next_states):
            max_q[i] = next_q[i, :len(state), :].amax(dim=(0, 1), keepdim=False)        # select non-padded section
        terminal_mask = ~torch.tensor(batch.is_terminals, dtype=torch.bool).to(device)
        rewards = torch.tensor(batch.rewards, dtype=torch.float32).to(device)
        return rewards + (terminal_mask * self.gamma * max_q).unsqueeze(1)

    def learn(self, batch: BatchSample) -> torch.Tensor:
        q_values, category_probs = self.q_model(batch.states).to(device)
        batch_size, seq_size, _ = q_values.shape
        b_indexes = torch.arange(batch_size).unsqueeze(1)
        s_indexes = torch.arange(seq_size)
        padded_actions = stack_padding(batch.actions, dtype="int32")
        q_predictions = q_values[b_indexes, s_indexes, padded_actions]
        # Mask to set padded q_values to zero
        mask = torch.zeros(batch_size, seq_size, dtype=torch.int32).to(device)
        for i, state in enumerate(batch.states):
            num_tokens = len(state)
            mask[i, :num_tokens] = 1
        # Train labels classifier
        q_predictions = (mask*q_predictions).sum(-1).div(mask.sum(-1))   # Mean q_prediction per sequence length
        q_targets = self.get_targets(batch)
        mse_loss = self.mse_criterion(q_predictions, q_targets).mean()
        # Train categories classifier
        target_categories = torch.from_numpy(self.encoded_labels[padded_actions, :])
        ce_loss = self.ce_criterion(category_probs, target_categories)                # TODO: Check logits vs probs
        ce_loss = ce_loss.sum(-1).div(mask.sum(-1))
        ce_loss = ce_loss.mean()
        # Compute total loss
        loss = mse_loss + ce_loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss          # TODO: Return ce and mse losses

    def update_target(self, tau: float = 1.0):
        layer_pairs = (
            (self.q_model.labels_classifier, self.tq_model.labels_classifier),
            (self.q_model.category_classifier, self.tq_model.category_classifier),
        )
        for (src_classifier, trg_classifier) in layer_pairs:
            src_params = src_classifier.named_parameters()
            trg_params = trg_classifier.named_parameters()

            # TODO: Parallelize this
            param_dict = {}
            for ((name, src_value), (_, trg_value)) in zip(src_params, trg_params):
                param_dict[name] = tau * src_value.data + (1 - tau) * trg_value.data
            trg_classifier.load_state_dict(param_dict)
