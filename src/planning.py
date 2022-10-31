import torch
import numpy as np

from src.utils import apply_labels_at, get_lev_dist


def get_lev_dist_of_next_tokens(tokens, reference, tok_i, tok_label):
    new_tokens = apply_labels_at(tokens, [tok_label], [tok_i])
    lev_dist = get_lev_dist(new_tokens, reference)
    return lev_dist


def get_best_candidates(mask_gen, tokens, reference, tok_i, tok_edit, src_ref_lev_dist, verbose=False):
    tok_labels = mask_gen.edit_to_labels(tok_edit)
    distances = np.array([get_lev_dist_of_next_tokens(tokens, reference, tok_i, tok_label) for tok_label in tok_labels])
    candidate_mask = distances < src_ref_lev_dist
    candidate_labels = tok_labels[candidate_mask]
    if verbose:
        print(f"Candidate labels for the token `{tokens[tok_i]}` at index `{tok_i}`: {candidate_labels}")
    return candidate_labels


def get_best_label(policy, mask_gen, tokens, tok_i, candidate_labels, explore=True):
    num_candidates = len(candidate_labels)
    if num_candidates == 0:
        best_tok_label = "$KEEP"
    elif num_candidates == 1:
        best_tok_label = candidate_labels[0]
    else:
        candidate_indexes = mask_gen.labels_to_actions(list(candidate_labels))
        if explore:
            candidate_probs = None            # Use uniform sampling
        else:
            with torch.no_grad():
                [logits] = policy([tokens])
                candidate_probs = logits[tok_i, candidate_indexes].softmax(0).cpu().numpy()
        best_tok_label = np.random.choice(candidate_labels, p=candidate_probs)
    return best_tok_label


def search_best_actions(policy, tokens, mask_gen, reference, explore=True, verbose=False):
    # Search labels
    num_tokens = len(tokens)
    labels = ["$KEEP"]*num_tokens
    if tokens != reference:
        current_lev_dist = get_lev_dist(tokens, reference)
        edit_mask = mask_gen.get_edit_mask(tokens, [reference])
        if verbose:
            print(f"Edit_mask: {edit_mask}")
        for tok_i in reversed(range(num_tokens)):
            tok_edit = edit_mask[tok_i]
            if tok_edit == "equal":
                continue
            elif tok_edit == "delete":
                tok_label = "$DELETE"
            else:
                candidate_labels = get_best_candidates(mask_gen, tokens, reference, tok_i, tok_edit, current_lev_dist, verbose=False)
                tok_label = get_best_label(policy, mask_gen, tokens, tok_i, candidate_labels, explore=explore)
            tokens = apply_labels_at(tokens, [tok_label], [tok_i])
            current_lev_dist = get_lev_dist(tokens, reference)
            labels[tok_i] = tok_label
    actions = mask_gen.labels_to_actions(labels)
    return actions
