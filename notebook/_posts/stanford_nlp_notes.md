---
title: "Stanford NLP Course Notes"
date: August, 2019
comments: true
author: "Jonathan Ramkissoon"
math: true
---


## Notes Taken from Stanford NLP Course

The main idea of word2vec is to predict every word and its context words. It achieves this using two algorithms:

-   Skip-gram: Predict context words given a target (position independent)
-   Continuous Bag-of-Words: Predict target words from bag-of-words

**Idea of the Skip-Gram**

We want to take one word as the "target" or "center" and predict the words around it, where around is determined by a pre-specified window size. This is essentially predicting the context of the target word. The model will assign a probability to all words in the vocabulary, given the target word.

For each word, $t = 1..T$, predict surrounding words in a window of "radius" $m$ around every word.

> "The dog is very big"
>
> -   Target - Very
> -   Window size - 1
> -   Context words - "is", "big"

**Objective Function**

Maximize the probability of any context word, given the current target word.

$$
J'(\theta) = \prod_{t = 1}^{T} \prod_{-m \le j \le m, j \ne 0} p(w_{t + j} | w_t; \theta)
$$

Therefore, we want to minimize the negative log-likelihood:

$$
J(\theta) = -\frac{1}{T} \sum^{T}_{t = 1} \sum_{-m \le j \le m, j \ne 0} log(p(w_{t+j} | w_t; \theta))
$$

The only parameters in this model are $\\theta$, which are the vector representations of each word in the vocabulary, and the window size.

The simplest representation of $p(w_{t+j} | w_t; \theta))$ is:

$$
\frac{e^{u_o^T v_c}}{\sum_{w=1}^V e^{u_w^T v_c}}
$$

Where:

-   $t$ and $j$ are positions in the input text
-   $c$ and $o$are indicies of the words at positions $t$ and $j$ in the vocabulary
-   $u_o$ is the contextual vector representation of word at index $o$ in the vocabulary. $u_o = w_{t+j}$
-   $v_c$ is the target representation of the word at index $c$ in the vocabilary $v_c = w_t$

**Note on Softmax** - We take the dot product of our 2 word vectors as a measure of similarity. If the contents of the vectors are similar to each other, the procut will get bigger. Then softmax turns this product into a probability distribution over all possible context words. It's called softmax because we take exponents before dividing by the sum, which will increase the large original values disproportionately.

**Training the Model**

We take all the parameters of the model into a vector, $\theta$. For each word, we'll have a $d$-dimensional vector for a context word, and a $d$-dimensional vector for a center (target) word. Therefore, $\theta \in R^{2dV}$ for $V$ words in the vocabulary.

To train the model, we have to find the gradients of the loss function.

**Negative Sampling**

The main idea behind negative sampling is to train binary logistic regressions for a true pair versus a couple of noise pairs. i.e. target and context word VS target and random word.

The overall objective function is:

$$
J(\theta) = \frac{-1}{T} \sum_{t=1}^{T}J_t(\theta)
$$

$$
J_t(\theta) = log(\sigma(u_o^T v_c)) + \sum_{i = 1}^{k}E_{j \sim P(w)} [log(\sigma(-u_j^T v_c))]
$$

$$
j \sim P(w) = U(w)^{3/4}/Z
$$

It'll come in handy to note that:

$$
log(\sigma(-u_j^T v_c)) = log(1 - \sigma(u_j^T v_c))
$$

What's actually going on in these equations?

-   $u_o$: Vector representation of word $o$ in the vocabulary as a contextual word
-   $v_c$: Vector representation of word $c$ in the vocabulary as a target word
-   $T$: Each window in the corpus. This means that $J(\\theta)$ averages the value of $J_t(\\theta)$ for each window.
-   $log(\\sigma(u_o^Tv_c))$: This is the first part of the objective function. It can be interpreted as the log "probability" of the target and context words co-occurring. The word probability is used loosely here, and is only thrown around because the sigmoid funciton returns a number between 0 and 1. We want to maximize this probability.
-   $\\sum_{i = 1}^{k}E_{j \\sim P(w)} [log(\sigma(-u_j^T v_c))]$: Here we sample $k$ words from the corpus according to a sampling distribution $P(w)$. For each of these sampled words, $j$, we calculate the log probability of $j$ not occuring in the context of the target word, $c$. In other words, we are trying to minimize the probability that this randomly sampled word, $j$, occurs in context of the target word, $c$.
-   $P(w) = U(w)^{3/4}/Z$: This is the sampling distribution, $U(w)$ is the unigram distribution. It is raised to the $3/4$th power to sample rarer words in the vocabulary, as opposed to words like "the" and "and".

Summary of Word2Vec:

-   Iterate through every word in the whole corpus
-   Predict surrounding words (outside words) using word vectors
-   Probability of the word, $o$ being an outside word of center word $c$. $p(o|c) = \frac{e^{u_o^T v_c}}{\\sum_{w=1}^V e^{u_w^T v_c}}$
-   Update these vectors so we can predict better
-   We have 2 matrices, one for center words and one for outside words. The vector representation of each word is a row, as opposed to a column. When we do our computations, we take a dot product of a word, say $v_4$ and all the outside words, $U$. This is $U.v_4^T$, which gives us a vector of dot product scores, which we run element wise softmax on each. We predict the same probability distribution for each position in the window, there's no granularity of prediction position. We want a model that gives reasonably high probability estimates to all words that occur in the context of the center word.
-   We have a problem with high frequency words like "the", "and", "of".
