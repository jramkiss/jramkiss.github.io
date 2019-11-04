---
layout: post
title: "One-Page Wonder: ULMFiT"
date: 2019-11-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: All the information you need to know to understand ULMFiT without spending 1 week on it
---

## Overview
To set the scene for the rest of this article, lets assume that the task is to build a classifier that can determine whether poems are happy or sad.


## Stages in Training a ULMFiT Model
#### Training Language Model on General Data

This is the most expensive training stage, where we teach a model about the general structure of language eg: a sentence has a subject-verb-object. In ULMFiT, the model architecture used is an AWD-LSTM (think regular LSTM with dropout on steroids) and the data used is from Wikipedia articles. Embeddings for each word in the corpus are learnt in this stage.

**How is this done?**

#### Fine-Tuning Language Model on Target Data

Usually in practical NLP tasks, the target data varies slightly from the data the LM was trained on. For this reason, we need to expose the LM to our target data so it can learn its structure and any new words that weren't in the general data. In our example task, the structure of poems is distinctly different from Wikipedia.

###### Variable Length Backpropagation Through Time

Backpropagation through time (BPTT) is the algorithm used to update the weights of an RNN. Errors are calculated and accumulated for each token in the sequence and backpropagated (update network weights) when we've reached a pre-specified number of tokens. The problem here is that we show the model the same sequence every epoch. In Variable Length BPTT, we randomize the number of tokens to be processed before backpropagating. The actual length is sampled from a normal distribution, who's mean is $x$ with 0.95 probability and $\frac{x}{2}$ otherwise.

#### Building the Classifier



## Resources
- [Understanding AWD-LSTM](https://yashuseth.blog/2018/09/12/awd-lstm-explanation-understanding-language-model/)
- [ULMFiT State of the Art in Text Analysis](https://humboldt-wi.github.io/blog/research/information_systems_1819/group4_ulmfit/)
- [Introduction to Backpropagation Through Time](https://machinelearningmastery.com/gentle-introduction-backpropagation-time/)
