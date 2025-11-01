import torch
def positional_encoding(seq_length, d, n=10_000):
    P = torch.zeros((seq_length, d)) # Initiate the matrix

    pos = torch.arange(seq_length).unsqueeze(1) # Get all positions
    dim_idx = torch.arange(0, d, 2) # Get all dimensions

    denominator = torch.pow(n, dim_idx/d) # Calculate the denominator (n^(2i/d))
    P[:, 0::2] = torch.sin(pos/denominator) # Get the positional encoding for all even positions
    P[:, 1::2] = torch.cos(pos/denominator) # Get the positional encoding for all odd positions
    return P
