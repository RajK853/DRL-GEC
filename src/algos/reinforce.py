import math
import torch
from torch.distributions import Categorical

from src.utils import stack_padding


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_log_pi(policy, states, actions):
    logits = policy(states)
    dist = Categorical(logits=logits)
    actions = stack_padding(actions, dtype="float32")
    actions = torch.from_numpy(actions).to(logits.device)
    # Obtain mask of the non-padded tokens
    batch_size, seq_len, label_size = logits.shape
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=logits.device)
    for i, state in enumerate(states):
        mask[i, :len(state)] = True
    log_pis = dist.log_prob(actions)
    log_pis = log_pis*mask                                    # Set log_pis of padded tokens to zero
    return log_pis


@torch.cuda.amp.autocast()
def compute_loss(policy, state_batch, action_batch, return_batch):
    log_pis = get_log_pi(policy, state_batch, action_batch)
    seq_log_pis = log_pis.sum(-1)                               # Sum log_probs over tokens
    pi_loss = -(seq_log_pis * return_batch).mean()
    return pi_loss


def train(pbar, optim, grad_scaler, policy, buffer, batch_size=32):
    num_items = len(buffer["state"])
    accumulation_size = math.ceil(num_items/batch_size)
    # Set up the progress bar
    pbar.reset()
    pbar.total = num_items
    buffer["return"] = torch.tensor(buffer["return"], device=device)
    losses = []
    optim.zero_grad()
    for i in range(0, num_items, batch_size):
        state_batch = buffer["state"][i:i+batch_size]
        action_batch = buffer["action"][i:i+batch_size]
        return_batch = buffer["return"][i:i+batch_size]
        pi_loss = compute_loss(policy, state_batch, action_batch, return_batch)
        grad_scaler.scale(pi_loss/accumulation_size).backward()
        losses.append(pi_loss)
        pbar.update(len(state_batch))
    grad_scaler.step(optim)
    grad_scaler.update()
    pbar.refresh()
    return torch.stack(losses).mean().item()
