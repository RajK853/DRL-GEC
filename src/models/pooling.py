import torch


def reduce_last(tensor, align):
    batch_size, _, embed_size = tensor.shape
    masks = (align != 0).unsqueeze(2)
    # Select tensor indexes
    batch_indexes = torch.arange(batch_size).view(-1, 1, 1)
    seq_indexes = (align.cumsum(axis=1) - 1).unsqueeze(2)     # -1 to match zero-indexing, unsqueeze to match the shape
    embed_indexes = torch.arange(embed_size)
    return masks*tensor[batch_indexes, seq_indexes, embed_indexes]


def reduce_mean(tensor, align):
    batch_size, _, encode_size = tensor.shape
    _, seq_size = align.shape
    tensor_out = torch.zeros(batch_size, seq_size, encode_size, dtype=torch.float32, device=tensor.device)
    for batch_i in range(batch_size):
        seq_offset = 0
        for seq_i in range(seq_size):
            alignment = align[batch_i, seq_i]
            if alignment == 0:
                break                                            # Stop if padding token alignment
            else:
                start_i = seq_i + seq_offset
                end_i = start_i + alignment
                tensor_out[batch_i, seq_i, :] = tensor[batch_i, start_i:end_i, :].mean(dim=0)
            seq_offset += alignment-1                            # -1 as seq_i increases by 1 in every iteration
    return tensor_out


def reduce_sum(tensor, align):
    batch_size, _, encode_size = tensor.shape
    _, seq_size = align.shape
    tensor_out = torch.zeros(batch_size, seq_size, encode_size, dtype=torch.float32, device=tensor.device)
    for batch_i in range(batch_size):
        seq_offset = 0
        for seq_i in range(seq_size):
            alignment = align[batch_i, seq_i]
            if alignment == 0:
                break                                            # Stop if padding token alignment
            else:
                start_i = seq_i + seq_offset
                end_i = start_i + alignment
                tensor_out[batch_i, seq_i, :] = tensor[batch_i, start_i:end_i, :].sum(dim=0)
            seq_offset += alignment-1                            # -1 as seq_i increases by 1 in every iteration
    return tensor_out
