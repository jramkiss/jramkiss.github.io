---
layout: post
title: "Latent Dirichlet Allocation"
date: 2023-02-11 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: An introduction to the math of latent Dirichlet allocation 
---


Latent Dirichlet allocation (LDA) has been the de-facto method for topic modeling over the last decade (maybe 2 decades). This post walks through the math behind how LDA works. I use the same notation as the [original paper](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) for consistency:

- The vocavulary consists of $V$ words which are one-hot encoded vectors in the model
- A document, $\boldsymbol{w}$, is a sequence of $N$ words, $(w_1, w_2, \dots, w_N)$
- The corpus, $D$, is a collection of $M$ documents 

<!-- Starting by decomposing the name, which may sound intimidating, "latent" just means hidden, so we are trying to find hidden topics in a corpus of text. "Dirichlet" is a discrete distribution of distributions. I have another post explaining this, [here](/_posts/2020-05-08-beta-and-dirichlet-distributions.md). -->

## Introduction 

LDA is a generative probabilistic model of documents in a corpus. The easiest way to conceptualize LDA is to consider how it "thinks" about documents, i.e. how it generates documents. I will first explain a simple version of this process in plane words, then introduce notation and math.

1) Choose a document length, $N$
2) Choose a discrete distribution of topics for that document. This can be thought of as a weighted (normalized) list of topics that summarize the document. For example, if we have [this](https://www.wired.com/story/openai-dalle-copyright-intellectual-property-art/) article, a possible topic distribution can be 50% "AI", 50% "Art" and 0% for all other topics 
3) Start creating the document word-by-word as: 
   1) Choose a single topic from the topic distribution above
   2) Choose a word conditional on the chosen topic 

This generative process makes it clear that LDA is a hierarchical model. 

## Probabilistic Model 

And here is some more notation that will be used throughout the post, but has already been introduced by the generative algorithm in the introduction:

- The corpus is considered to have $K$ topics which is a hyperparameter that must be set in advance
- The random variable representing the distribution of distribution of topics for a given document is $\theta \in \mathbb{R}^K$. This is not a typo and will become clear later and is the variable of importance to us
- A single topic from the topic distribution, $\theta$, is given by $z$. Each word is associated with a topic, so the word $w_n$ is associated with topic $z_n$

With this notation, the algorithm above can be re-written as: 

1) Sample a document length, $N \sim p(N)$. This is taken to be $\text{Poi}(\nu)$ in the original paper, but the choice of prior here is of little consequence
2) Sample a topic distribution, $\theta \sim \text{Dir}(\alpha)$
3) For each of the $N$ words: 
   1) Sample a topic, $z_i \sim \text{Multinom}(\theta)$
   2) Sample a word, $w_i \sim p(w_i \mid z_i, \beta)$

Where the distribution of words is conditional a matrix of conditional probabilities, $\beta \in \mathbb{R}^{K \times V}$, which are parameters of the model. Each entry in this matrix, $\beta_{ij} = p(w_j = 1 \mid z_i = 1)$, represents the probability that word $j$ occurs given that we have selected topic $i$. 

#### Sidenote: Distribution of Distributions 

In many cases, the Dirichlet distribution is described as a distribution of distributions. This used to give me headaches and felt like a circular definition, which is the reason for this sidenote. A $K$-dimensional Dirichlet distribution is a distribution over the $(K-1)$-simplex. If $\theta \sim \text{Dir}_K(\alpha)$, then for $i=1, \dots, K, \theta_i \ge 0$ and $\sum_i \theta_i = 1$. In other words, a sample from a K-dimensional Dirichlet distribution is a K-length vector where each element is $\ge$ zero and the sum of the vector is 1, which is the definition of a dicrete distribution. This is where the phrase "distribution over distributions" comes from. 

<!-- We can then specify the joint distribution of a topic mixture with a set of $N$ topics and words. Note that this is not the likelihood function, since it is for one document only **confirm this**:

$$
p(\theta, \boldsymbol z, \boldsymbol w \mid \alpha, \beta) = p(\theta \mid \alpha) \prod_{i=1}^N p(z_i \mid \theta) p(w_i \mid z_i, \beta)
$$ -->

## Graphical Representation 

<div class='figure' align="center">
    <img src="/assets/LDA_pgm.png" width="65%" height="65%">
    <div class='caption' width="70%" height="70%">
        <p> Plate notation for LDA, taken from the original paper. </p>
    </div>
</div>

From the plate notation, $(\alpha, \beta)$ are generated once for the entire corpus. Following this, a new topic distribution, $\theta$ is generated for each document, resulting in $M$ total samples. For each of these samples $z$ and $w$ are generated $N$ times per document, conditional on $\theta$. The sampling of the $\theta$ for each document allows documents to be associated with more than one topic. 

This approach is more flexible than a Dirichlet-Multinomial mixture model, where the Dirichlet distribution is sampled once for the corpus, then the document topic is generated by a Multinomial conditioned on the Dirichlet sample. Not only is this less flexible from the standpoint of a traditional hierarchical model, it also restricts documents to only have one topic. By using $\alpha, \beta$ as global variables (sampled once per corpus) and conditioning on them, documents are allowed to have a distribution of topics. 

