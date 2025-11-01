import torch
from torch import nn

from .encoder_block import EncoderBlock
from .decoder_block import DecoderBlock


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, num_heads, embed_size, num_layers=1):
        super().__init__()
        self.enc_layer = nn.ModuleList([
            EncoderBlock(num_heads=num_heads, embed_size=embed_size, dropout=0.1, expansion_factor=4)
            for _ in range(num_layers)
        ])
        self.dec_layer = nn.ModuleList([
            DecoderBlock(num_heads=num_heads, embed_size=embed_size, dropout=0.1, expansion_factor=4)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(in_features=embed_size, out_features=tgt_vocab_size)
        self.encoder_embeddings = nn.Embedding(embedding_dim=embed_size, num_embeddings=src_vocab_size)
        self.decoder_embeddings = nn.Embedding(embedding_dim=embed_size, num_embeddings=tgt_vocab_size)
        self.embed_size = embed_size

    def positional_encoding(self, seq_length, d, n=10_000):
        P = torch.zeros((seq_length, d)) # Initiate the matrix

        pos = torch.arange(seq_length).unsqueeze(1) # Get all positions
        dim_idx = torch.arange(0, d, 2) # Get all dimensions

        denominator = torch.pow(n, dim_idx/d) # Calculate the denominator (n^(2i/d))
        P[:, 0::2] = torch.sin(pos/denominator) # Get the positional encoding for all even positions
        P[:, 1::2] = torch.cos(pos/denominator) # Get the positional encoding for all odd positions
        return P

    def forward(self, encoder_input, decoder_input, mask):
        batch_size, src_seq_len = encoder_input.shape # Shape of encoder's input
        _, tgt_seq_len = decoder_input.shape # Shape of decoder's input

        # Encoder Block
        src_P = self.positional_encoding(seq_length=src_seq_len, d=self.embed_size) # Positional Encoding
        encoder_embeddings = self.encoder_embeddings(encoder_input) # Trainable Word Embeddings
        encoder_full_embeddings = encoder_embeddings + src_P # Final Word Embeddings (Word Embeddings + Positional Encoding)
        encoder_output = encoder_full_embeddings
        for layer in self.enc_layer:  # Fixed variable name
            encoder_output = layer(encoder_output)

        # Decoder Block
        trgt_P = self.positional_encoding(seq_length=tgt_seq_len, d=self.embed_size) # Positional Encoding
        decoder_embeddings = self.decoder_embeddings(decoder_input) # Trainable Word Embeddings
        decoder_full_embeddings = decoder_embeddings + trgt_P # Final Word Embeddings (Word Embeddings + Positional Encoding)
        decoder_output = decoder_full_embeddings
        for layer in self.dec_layer:  # Fixed variable name
            decoder_output = layer(decoder_output, mask, encoder_output)

        # Output
        fc_output = self.fc_out(decoder_output)
        return fc_output
