import torch
import numpy as np

from src.utils import apply_labels_at, get_lev_dist, softmax


def get_lev_dist_of_next_tokens(tokens, reference, tok_i, tok_label):
    new_tokens = apply_labels_at(tokens, [tok_label], [tok_i])
    lev_dist = get_lev_dist(new_tokens, reference)
    return lev_dist


def get_best_candidates(mask_gen, tokens, reference, tok_i, tok_edit, src_ref_lev_dist, verbose=False):
    tok_labels = mask_gen.edit_to_labels(tok_edit)
    distances = np.array([get_lev_dist_of_next_tokens(tokens, reference, tok_i, tok_label) for tok_label in tok_labels])
    delta = distances - src_ref_lev_dist
    candidate_mask = delta < 0
    candidate_labels = tok_labels[candidate_mask]
    candidate_delta = np.abs(delta[candidate_mask])     # Ignore sign as we are only interested in magnitude
    if verbose:
        print(f"Candidate labels for the token `{tokens[tok_i]}` at index `{tok_i}`: {candidate_labels}")
    return candidate_labels, candidate_delta


def get_best_label(policy, mask_gen, tokens, tok_i, candidate_labels, candidate_delta, explore=True):
    num_candidates = len(candidate_labels)
    if num_candidates == 0:
        return "$KEEP"
    # Insert $KEEP label to give chance to not modify the token
    candidate_labels = np.append(candidate_labels, "$KEEP")
    candidate_delta = np.append(candidate_delta, 0)
    if explore:
        candidate_probs = None
    else:
        with torch.no_grad():
            [logits] = policy([tokens])
            candidate_indexes = mask_gen.labels_to_actions(list(candidate_labels))
            policy_probs = logits[tok_i, candidate_indexes].softmax(0).cpu().numpy()
            delta_probs = softmax(candidate_delta)
            candidate_probs = 0.5*(policy_probs + delta_probs)  # Calculate elementwise mean between two probabilities
            candidate_probs /= candidate_probs.sum()    # Normalize to make sure sum(probs) is exactly 1.0
    best_tok_label = np.random.choice(candidate_labels, p=candidate_probs)
    return best_tok_label


def search_best_actions(policy, tokens, reference, mask_gen, explore=True, verbose=False):
    # Search labels
    num_tokens = len(tokens)
    labels = ["$KEEP"]*num_tokens
    if tokens != reference:
        current_lev_dist = get_lev_dist(tokens, reference)
        edit_mask = mask_gen.get_edit_mask(tokens, [reference])
        if verbose:
            print(f"Edit_mask: {edit_mask}")
        for tok_i, tok_edit in reversed(tuple(enumerate(edit_mask))):
            if tok_edit == "equal":
                continue
            candidate_labels, candidate_delta = get_best_candidates(
                    mask_gen, tokens, reference, tok_i, tok_edit, current_lev_dist, verbose=verbose
            )
            tok_label = get_best_label(
                    policy, mask_gen, tokens, tok_i, candidate_labels, candidate_delta, explore=explore
            )
            if tok_label != "$KEEP":
                tokens = apply_labels_at(tokens, [tok_label], [tok_i])
                current_lev_dist = get_lev_dist(tokens, reference)
                labels[tok_i] = tok_label
    actions = mask_gen.labels_to_actions(labels)
    return actions
