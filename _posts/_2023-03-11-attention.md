---
layout: post
title: "TL;DR on Attention"
date: 2023-04-21 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: 
---

When I first looked at attention a couple years back, somehow the naming scheme of "queries", "keys" and "values" was more confusing than just looking at the math. Rare occurrence. Nevertheless, I have spent some time understanding attention and wanted to write a quick tl;dr. 

There are many different types of attention used in language models these days, but they are all variants of the same basic idea, which this post describes. The goal of the attention mechanism is to find a representation of an input that incorporates some context. This is intentionally vague, but the typical example is with text modelling. For example, "apple" has different meanings depending on is we're talking about the food or the company. The attention mechanism is able to combine the input and context by using a series of transformations, parameterized by 3 matrices: $W_Q, W_K, W_V$. 

To see how this is done, consider the simple case where we want to find the contextual embedding for one token, $X \in \mathbb{R}^{1 \times d_x}$, with a context sequence, $Z \in \mathbb{R}^{n \times d_z}$. Following from the example above, $X$ is the embedding for the word "apple" and $Z$ is a matrix of embeddings for the sentence "We are eating breakfast". The parameter matrices of the attention head are:

- $W_Q \in \mathbb{R}^{d_x \times d_q}$
-  $W_K \in \mathbb{R}^{d_z \times d_q}$ 
-  $W_V \in \mathbb{R}^{d_z \times d}$

Where $d$ is the size of the final contextual embedding. To combine the input and context, the input is mapped to a "query", as: $Q = X W_Q \in \mathbb{R}^{1 \times d_q}$ and the context to a "key": $K = Z W_K \in \mathbb{R}^{n \times d_q}$. For me, it was easier to discard these names and just think of them as intermediate vector representations (like hidden layers), but after some time the names stuck. The context sequence is then mapped to another matrix called the "value" matrix, as: $V = Z W_V \in \mathbb{R}^{n \times d}$. 

The next step is to combine the query and key matrices as $QK^T \in \mathbb{R}^{1 \times n}$. The resulting matrix is interpreted as unnormalized scores, representing the amount of weight (or attention) to put on each of the tokens in the context sequence. To find the final contextual embedding for $X$, we normalize these weights using softmax and find a weighted average of the value matrix, $V$ as: 

$$ X_{\text{contextual}} = \text{softmax}(QK^T) V \in \mathbb{R}^{1 \times d}$$

In practice, the unnormalized attention scores are scaled by a factor of $\sqrt{d_q}$ before applying softmax to control the variance, but for illustrative purposes I've left this out. 

In this example, the input "attends" to the context sequence, which is a different sequence. This is called cross-attention. In some applications, the input attends to itself, which is self-attention. 


## Attention with PyTorch 

Here is a bare-bones implementation of attention using PyTorch:

```python
class Attention (nn.Module):
    """
    Implementation of self/cross bi-directional/uni-directional attention. To use this for self attention, 
    pass the same input and context size on initialization. To use bi-directional/uni-directional 
    attention, pass different masks to the forward method
    
    Args: 
        - X: sequence to find contextual representation of, \in \mathcal{R}^{d_{x}, l}
        - Z: sequence to attent to, \in \mathcal{R}^{d_{z}, l}
    
    Returns: 
        - X_contextual: Updated embeddings of tokens in X with information from tokens in Z incorporated 
    """
    
    def __init__ (self, inp_embd_size, context_embd_size, attn_embd_size, out_embd_size, **kwargs):
        super(Attention, self).__init__()
        self.attn_embd_size = attn_embd_size
        self.out_embd_size = out_embd_size
        
        self.W_q = nn.Linear(in_features=inp_embd_size, out_features=self.attn_embd_size) # query embedding 
        self.W_k = nn.Linear(in_features=context_embd_size, out_features=self.attn_embd_size) # key embedding
        self.W_v = nn.Linear(in_features=context_embd_size, out_features=self.out_embd_size)
        
    def forward (self, primary_seq, context_seq, attn_mask, **kwargs):
        """
        Args: 
            - primary_seq: Sequence to find contextual embeddings of. Shape (inp_embd_size, context_maxlen)
            - context_seq: Contextual sequence of shape (context_embd_size, context_maxlen)
            - attn_mask: Determines which of the tokens in the context to be masked
        Q = W_q @ X
        K = W_k @ Z
        V = W_v @ Z
        X_contextual = softmax(QK^T / sqrt(attn_embd_size)) @ V
        """
        Q = self.W_q(X.T)
        K = self.W_k(Z.T)
        V = self.W_v(Z.T)
        
        unnormalized_weights = Q @ K.T / self.attn_embd_size**0.5
        unnormalized_weights[:, attn_mask == 0] = -1e10
        
        attn_weights = F.softmax(unnormalized_weights, dim=0)
        contextual_embd = attn_weights @ V
        
        return contextual_embd.T
```

```python 
inp_embd_size = 10
context_embd_size = 13
max_len = 5

sa = Attention(inp_embd_size=inp_embd_size,
               context_embd_size=context_embd_size,
               attn_embd_size=15,
               out_embd_size=25)

X = torch.rand((inp_embd_size, max_len))
Z = torch.rand((context_embd_size, 7))
```


