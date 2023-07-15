---
layout: post
title: "Transformer Exercises Solutions"
date: 2023-04-02 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Solutions to the transformer exercises posted in [Circuits](https://transformer-circuits.pub/)
---

These are my answers to the transformer exercises posted in [Circuits](https://transformer-circuits.pub/2021/exercises/index.html).

# Questions 

**1: Describe the transformer architecture at a high level**

Transformers are neural network models that use attention to better model dependence in sequences. There are different types of transformers, which differ based on the training task and type of attention used. Here are a few: 
    1. Encoder-decoder transformer: This is mainly used in seq-to-seq tasks, such as machine translation. The goal is to translate an input/context sequence, $z_n$, to an output/primary sequence, $x_n$. The model uses attention in both the encoding and decoding layers to produce embeddings for sequences. First, the context sequence, $z_n$, is encoded using bidirectional multi-head attention. Then, the output/primary sequence, $x_n$, is encoded using cross-attention and masked unidirectional self-attention, with the right side of each token is masked. 
    2. Encoder/decoder only transformers: These are 2 different types of transformers that differ slightly in their attention masks. The encoder only transformer does not mask any tokens, so each token can attent to all others in the sequence, this is bidirecetional attention. Whereas the decoder only transformer uses unidirectional attention, where tokens can only attend to previously "visited" tokens in the sequence, i.e. tokens to the left of it. "Visited" is in quotes because the attention mechanism is not applied sequentially, but to all tokens in parallel. BERT-style models use encoder only transformers, while GPT-style models use decoder only transformers. 

**2: Describe how an individual attention head works in detail, in terms of the matrices $W_Q$, $W_K$, $W_V$ and $W_{out}$. (The equations and code for an attention head are often written for all attention heads in a layer concatenated together at once. This implementation is more computationally efficient, but harder to reason about, so we'd like to describe a single attention head.)**


The goal of the attention mechanism is to find a new representation of a primary sequence given the context sequence. This is done by encoding each sequence and combining these representations such that the representation of each token in the primary sequence is a weighted sum of tokens in the context sequence. To see how this is done, consider the simple case where we want to find the contextual embedding for one token, $X \in \mathbb{R}^{1 \times d_x}$, given a context sequence, $Z \in \mathbb{R}^{n \times d_z}$. An example of this can be finding the contextual embedding of the token: "apple", with the context sequence "We are eating breakfast". The attention head comprises of 3 parameter matrices, $W_Q \in \mathbb{R}^{d_x \times d_q}$, $W_K \in \mathbb{R}^{d_z \times d_q}$ and $W_V \in \mathbb{R}^{d_z \times d}$, where $d$ is the size of the final contextual embedding. First, we map the primary sequence into a "query", as: $Q = X W_Q \in \mathbb{R}^{1 \times d_q}$. The same is done for the context sequence to find the "key" vector: $K = Z W_K \in \mathbb{R}^{n \times d_q}$. Finally, the context sequence is mapped to another matrix called the "value" matrix, as: $V = Z W_V \in \mathbb{R}^{n \times d}$. The next step is to combine the queries (in this case we only have 1) and keys, which is done with matrix multiplication: $QK^T \in \mathbb{R}^{1 \times n}$. These are now unnormalized scores, representing the amount of weight (or attention) to put on each of the tokens in the context sequence. To find the final contextual embedding, we normalize these weights using softmax and find a weighted average of the value matrix as: 

$$ X_{\text{contextual}} = \text{softmax}(QK^T) V \in \mathbb{R}^{1 \times d}$$

In reality, the unnormalized attention scores are scaled by a factor of $\sqrt{d_q}$ before the softmax, but for illustrative purposes I've left this out. 


**3: Attention heads move information from a subspace of the residual stream of one token to a different subspace in the residual stream of another. Which matrix controls the subspace that gets read, and which matrix controls the subspace written to? What does their product mean?**



**4: Which tokens an attention head attends to is controlled by only two of the four matrices that define an attention head. Which two matrices are these?**



**5: Attention heads can be written in terms of two matrices instead of four, $W_Q^T \cdot W_K$ and $W_{out} \cdot W_V$. In the previous two questions, you gave interpretations to these matrices. Now write out an attention head with only reference to them. What is the rank of these matrices?**



**6: You'd like to understand whether an attention head is reading in the output of a previous attention head. What does $W_V^2 \cdot W_{out}^1$ tell you about this? What do the singular values tell you?**

---

# Exercise 1: Build a simple virtual attention head

Small transformers often have multiple attention heads which look at the previous token, but no attention heads which look at the token two previous. In this exercise, we'll see how two previous token heads can implement a small "virtual attention head" looking two tokens behind, without sacrificing a full attention head to the purpose.

Let's consider two attention heads, head 1 and head 2, which both attend to the previous token. Head 1 is in the first layer, head 2 is in the second layer. To make it easy to write out explicit matrices, we'll have the k, q, and v vectors of both heads be 4 dimensions and the residual stream be 16 dimensions.

- Write down $W_V^1$ and $W_{out}^1$ for head 1, such that the head copies dimensions 0-3 of its input to 8-11 in its output.
- Write down $W_V^2$ and $W_{out}^2$ for head 2, such that it copies 3 more dimensions of the previous token, and one dimension from two tokens ago (using a dimension written to by the previous head)
- Expand out $W_{net}^1 = W_{out}^1 \cdot W_V^1$ and $W_{net}^2 = W_{out}^2 \cdot W_V^2$. What do these matrices tell you?
- Expand out the following matrices: Two token copy: $W_{net}^2 \cdot W_{net}^1$. One token copy: $W_{net}^2 \cdot I_d + I_d \cdot W_{net}^1$
- **Observation**: When we think of an attention head normally, they need to dedicate all their capacity to one task. In this case, the two heads dedicated 7/8ths of their capacity to one task and 1/8th to another.