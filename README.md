# OG Transformers Implementation
A from-scratch implementation of the original Transformer architecture from the paper [Attention Is All You Need](https://arxiv.org/html/1706.03762v7)

---
## Project Overview
This project was a self-learning exercise, I’ve always believed that the best way to understand something is to build it myself. So, I decided to reconstruct the Transformer from the ground up to grasp how attention, residuals, and normalization actually work under the hood.

---
## Directory Structure
```
src/
├── decoder_block.py         			# Decoder block logic and helper functions
├── encoder_block.py         			# Encoder block logic
├── multihead_attention.py   			# Multi-Head Attention mechanism
├── positional_encoder.py    			# Positional encoding implementation
├── self_attention.py        			# Single self-attention head logic
└── transformer.py           			# Full Transformer architecture

test_cases/
└── test_transformer_component.py 		# Testing every component (Using PyTest)
```

---
## Supporting Articles, Papers, and Videos
* [Attention Is All You Need](https://arxiv.org/html/1706.03762v7) Original paper.
* [A Gentle but Practical Introduction to Transformers in Deep learning](https://vnaghshin.medium.com/a-gentle-but-practical-introduction-to-transformers-in-deep-learning-75e3fa3f8f68) Used for conceptual clarity, not for code.
* 3Blue1Brown Neural networks: [Chapter 5](https://youtu.be/wjZofJX0v4M?si=sw25VdcSoDoYzda3), [Chapter 6](https://youtu.be/eMlx5fFNoYc?si=y1U6JO51xKTvJRNK), and [Chapter 7](https://youtu.be/9-Jl0dxWQs8?si=ALCErhyNmu92Frzy) (I really do suggest going through the entire playlist if you're new to deep learning, it is an absolute treasure.)

---
Feel free to contact me via [LinkedIn](https://www.linkedin.com/in/marwan-mohammed1/) or my [Email](marwanmohammed056@gmail.com) if you have have any questions. And feel free to fork this repo if you have any edits you'd like to make. Finally, don't forget to star the repo!
