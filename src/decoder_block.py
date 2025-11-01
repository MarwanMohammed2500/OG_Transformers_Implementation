import torch
from torch import nn, optim
from .multihead_attention import MultiHeadAttention

def look_ahead_mask(size: int):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 1

def prepare_decoder_inputs(target_sequences, bos_token_id, eos_token_id):
    """
    Prepare shifted-right inputs for teacher forcing
    """
    # Remove <eos> from end, add <bos> to beginning
    decoder_input = torch.cat([
        torch.full((target_sequences.shape[0], 1), bos_token_id),  # Add <bos>
        target_sequences[:, :-1]  # Remove last token (<eos>)
    ], dim=1)
    return decoder_input

class DecoderBlock(nn.Module):
    def __init__(self, num_heads, embed_size, dropout=0.1, expansion_factor=4):
        super().__init__()
        self.masked_mha = MultiHeadAttention(num_heads=num_heads, embed_size=embed_size)
        self.cross_mha = MultiHeadAttention(num_heads=num_heads, embed_size=embed_size)

        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.layer_norm_3 = nn.LayerNorm(embed_size)

        self.ffnn = nn.Sequential(
            nn.Linear(in_features=embed_size, out_features=expansion_factor * embed_size),
            nn.ReLU(),
            nn.Linear(in_features=expansion_factor * embed_size, out_features=embed_size)
        )

        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)

    def forward(self, x, mask, encoder_output):
        x = self.layer_norm_1(x)
        x = x + self.dropout_1(self.masked_mha(x, x, x, mask=mask))
        x = self.layer_norm_2(x)
        x = x + self.dropout_2(self.cross_mha(x, encoder_output, encoder_output))
        x = x + self.dropout_3(self.ffnn(self.layer_norm_3(x)))
        return x
