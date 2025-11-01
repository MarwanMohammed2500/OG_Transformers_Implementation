from torch import nn
from .multihead_attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, num_heads, embed_size, dropout=0.1, expansion_factor=4):
        super().__init__()
        # Sub-Layer 1
        self.mha = MultiHeadAttention(num_heads, embed_size)
        self.dropout_1 = nn.Dropout(p=dropout)

        # LayerNorm 1
        self.layer_norm_1 = nn.LayerNorm(embed_size)

        # Sub-Layer 2
        self.ffnn = nn.Sequential(
            nn.Linear(in_features=embed_size, out_features=expansion_factor * embed_size),
            nn.ReLU(),
            nn.Linear(in_features=expansion_factor * embed_size, out_features=embed_size)
        )
        self.dropout_2 = nn.Dropout(p=dropout)

        # LayerNorm 2
        self.layer_norm_2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = self.layer_norm_1(x)
        x = x + self.dropout_1(self.mha(x, x, x)) # Sub-Layer 1: Multi-Head Attention
        x = x + self.dropout_2(self.ffnn(self.layer_norm_2(x))) # Sub-Layer 2: Feed-Forward Neural Network
        return x
