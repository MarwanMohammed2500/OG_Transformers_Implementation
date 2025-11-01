import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.q = nn.Linear(in_features=embedding_dim, out_features=embedding_dim) # Query (W_Q . X)
        self.k = nn.Linear(in_features=embedding_dim, out_features=embedding_dim) # Key (W_K . X)
        self.v = nn.Linear(in_features=embedding_dim, out_features=embedding_dim) # Value (W_V . X)

    def forward(self, x):
        Q = self.q(x) # Train the Query weights and get the Query
        K = self.k(x) # Train the Key weights and get the Key
        V = self.v(x) # Train the Value weights and get the Value

        # Calculate the numerator of the attention equation (the dot-product attention)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Get D_K (The dimension size of K and Q)
        d_k = K.size(-1)

        # Calculate the attention weights (Normalize the weights)
        attention_weights = F.softmax(scores/np.sqrt(d_k), dim=-1)

        # Calculate the attention (attention_weights . V)
        attention = torch.matmul(attention_weights, V)
        return attention_weights, attention
