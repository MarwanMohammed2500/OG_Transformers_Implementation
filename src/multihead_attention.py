import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_size):
        super().__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.head_dim = embed_size // num_heads # Each head should be of dimension embedding_size/number_of_heads
        self.q = nn.Linear(in_features=embed_size, out_features=embed_size) # Query (W_Q . X)
        self.k = nn.Linear(in_features=embed_size, out_features=embed_size) # Key (W_K . X)
        self.v = nn.Linear(in_features=embed_size, out_features=embed_size) # Value (W_V . X)
        self.num_heads = num_heads # Number of attenton heads

        # Output Fully Connected layer
        self.fc = nn.Linear(in_features=embed_size, out_features=embed_size)

    def forward(self, query, key, value, mask=None):
        B = query.size(0) # Batch Size
        T_query = query.size(1) # Sequence Length for Query
        T_key = key.size(1) # Sequence Length for Key
        Q = self.q(query).reshape(B, T_query, self.num_heads, self.head_dim).transpose(1, 2) # Train the Query weights and get the Query
        K = self.k(key).reshape(B, T_key, self.num_heads, self.head_dim).transpose(1, 2) # Train the Key weights and get the Key
        V = self.v(value).reshape(B, T_key, self.num_heads, self.head_dim).transpose(1, 2) # Train the Value weights and get the Value

        scores = torch.matmul(Q, K.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(1), float("-inf"))

        # Get D_K (The dimension size of K and Q)
        d_k = K.size(-1)

        # Calculate the attention weights (Normalize the weights)
        attention_weights = F.softmax(scores/np.sqrt(d_k), dim=-1)

        # Calculate the attention (attention_weights . V)
        attention = torch.matmul(attention_weights, V)
        output = attention.transpose(1, 2).contiguous().reshape(B, T_query, -1)
        return self.fc(output)
