############################# Import Components ########################################
from ..src.positional_encoder import positional_encoding                               #
from ..src.self_attention import SelfAttention                                         #
from ..src.multihead_attention import MultiHeadAttention                               #
from ..src.encoder_block import EncoderBlock                                           #
from ..src.decoder_block import look_ahead_mask, prepare_decoder_inputs, DecoderBlock  #
from ..src.transformer import Transformer                                              #
import torch                                                                           #
from torch import nn                                                                   #
########################################################################################

# ------------- Initialize data ------------- #
input_ids = torch.tensor([[2, 3, 1, 5, 4]]) # 5×1 (batch size * sequence length)
seq_length = input_ids.shape[1]*2

# ------------- Initialize Embeddings ------------- #
embeddings_dim = 512
embedding = nn.Embedding(embedding_dim=embeddings_dim, num_embeddings=seq_length)
embeddings = embedding(input_ids)

# ------------- Initialize Single Attention Head ------------- #
self_attention = SelfAttention(embedding_dim=embeddings_dim)

# ------------- Test Positional Encoding ------------- #
def test_positional_encoding():
    P = positional_encoding(seq_length=input_ids.shape[1], d=512)
    global final_word_embeddings
    final_word_embeddings = embeddings + P # Original Word Embedding + Positional Encoding
    print(f"Final Word Embeddings (Positionally Encoded):\n{final_word_embeddings}")

# ------------- Test self attention head ------------- #
def test_self_attention():
    attention_head = SelfAttention(embeddings_dim)

    attn_weights, single_head_output = attention_head(final_word_embeddings)  # Each row in the attention weights represents the attention weights of a word. and each row in the output is the vector representation of a word
    print(f"Attention Weights:\n{attn_weights}\n\nSingle Attention Head Output:\nsingle_head_output")

# ------------- Test Multihead Attention ------------- #
def test_multihead_attention():
    mha = MultiHeadAttention(num_heads=8, embed_size=embeddings_dim)

    mha_output = mha(final_word_embeddings, final_word_embeddings, final_word_embeddings) # Each row in the output is the vector representation of a word
    print(f"MultiHead Attention Output:\n{mha_output}")

# ------------- Test Encoder Block ------------- #
def test_encoder_block():
    enc_block = EncoderBlock(num_heads=8, embed_size=embeddings_dim)
    global encoder_block_output
    encoder_block_output = enc_block(final_word_embeddings)
    print(encoder_block_output)
    print(f"Encoder Block Output's Shape: {encoder_block_output.shape}")
    print(f"Embedding's Size: {embeddings_dim}")

# ------------- Test Look-Ahead Mask ------------- #
def test_look_ahead_mask():
    size = final_word_embeddings.shape[1]
    global mask
    mask = look_ahead_mask(size)
    print(f"Look Ahead Mask:\n{mask}")

# ------------- Test A Masked version of Multihead Attention ------------- #
def test_masked_mha():
    masked_mha = MultiHeadAttention(num_heads=8, embed_size=embeddings_dim)
    masked_mha_output = masked_mha(final_word_embeddings, final_word_embeddings, final_word_embeddings, mask) # Each row in the output is the vector representation of a word
    print("Masked MultiHead Attention Output:\n{masked_mha_output}")

# ------------- Test prepare_decoder_inputs ------------- #
def test_prepare_decoder_inputs():
    target_sequences = input_ids
    global decoder_input
    decoder_input = prepare_decoder_inputs(target_sequences, bos_token_id=1, eos_token_id=2)
    print(f"Decoder's Input:\n{decoder_input}")

# ------------- Test prepare_decoder_inputs ------------- #
def test_decoder_block():
    dec_block = DecoderBlock(num_heads=8, embed_size=embeddings_dim)
    decoder_block_output = dec_block(final_word_embeddings, mask, encoder_block_output)
    print(f"Decoder Block's Output:\n{decoder_block_output}")

# ------------- Test Transformer ------------- #
def test_transformer():
    transfomer = Transformer(src_vocab_size=10, tgt_vocab_size=10, num_heads=8, embed_size=embeddings_dim)
    transfomer_output = transfomer(encoder_input=input_ids, decoder_input=decoder_input, mask=mask)
    print(f"Transformer's Output:\n{transfomer_output}")
