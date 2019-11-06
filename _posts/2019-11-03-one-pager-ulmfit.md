---
layout: post
title: "One-Page Wonder: ULMFiT"
date: 2019-11-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: All the information you need to know to understand ULMFiT without spending 1 week on it
---

#### Note to the Reader
I'll assume that you have some prior exposure to ULMFiT, and possibly have already implemented it on a task. If not, [this](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb) notebook walks through how to use the model.

To set the scene for the rest of this article, lets assume that the task is to build a classifier that can determine whether poems are happy or sad.

ULMFiT stands for Universal Language Model Fine Tuning, and is a method that efficiently utilizes transfer learning for NLP tasks.


## Stages in Training a ULMFiT Model
#### Training Language Model on General Data

This is the most expensive training stage, where we teach a model about the general structure of language eg: a sentence has a subject-verb-object. In ULMFiT, the model architecture used is an AWD-LSTM (think regular LSTM with regularization on steroids) and the data used is from Wikipedia articles. Embeddings for each word in the corpus are learnt in this stage.

For every token in the sentence, the output of the model is a vector containing the probability that every word in the vocabulary appears as the next word in the sentence. So the model output for the 3rd word in the sentence is a vector of probabilities that for the 4th word.

Below is a quick overview of some tricks implemented by the AWD-LSTM to improve accuracy and efficiency:

##### Variable Length Backpropagation Through Time

Backpropagation through time (BPTT) is the algorithm used to update the weights of an RNN. Errors are calculated and accumulated for each token in the sequence and backpropagated (update network weights) when we've reached a pre-specified number of tokens. The problem here is that we show the model the same sequence every epoch. In Variable Length BPTT, we randomize the number of tokens to be processed before backpropagating. The actual length is sampled from a normal distribution, with a mean of $x$ with 0.95 probability and $\frac{x}{2}$ otherwise.

##### Dropout

TODO


#### Fine-Tuning Language Model on Target Data

Usually in practical NLP tasks, the target data varies slightly from the data the LM was trained on. For this reason, we need to expose the LM to our target data so it can learn its structure and any new words that weren't in the general data. In our example task, the structure of poems is distinctly different from Wikipedia.

The process of fine-tuning models is delicate and often leads to overfitting when datasets were small and even catastrophic forgetting, where the model forgets everything it's previously learnt.

##### Freezing

The original model is trained on 240k unique tokens, meaning the size of the embedding and decoding matrices is $(240000, 400)$. To deal with resource (memory and computational power) constraints, ULMFiT alters the size of these matrices to be the vocabulary of the target data (or up to a pre-specified amount). The process of changing the size of the embedding matrix presents difficulties when we start to fine tune, as part of our embedding matrix is untrained. If we start to train the whole model, we risk catastrophic forgetting, where our LSTM's lose all the information the learned in step 1. Instead, we freeze all the LSTM weights and fit the model for one cycle, only updating the embedding and decoding layers. Then we unfreeze the model weights and continue training.

##### Slanted Triangular Learning Rates

The learning rate starts off large to quickly converge to a suitable region, then decays slowly to fine-tune the weights.

##### Discriminative Fine-Tuning

The AWD-LSTM contains 3 stacked LSTMs, capturing more specific information about the language at each layer. The difference in processes can be addressed by using different learning rates for each LSTM.

#### Building the Classifier

The last stage in training a ULMFiT model is to build the actual classifier. We already have a model that understands general language and domain-specific language (poetry in this case)

##### Concat Pooling

##### Linear Decoder

##### Gradual Unfreezing 

## Resources
- [ULMFiT Original Paper](https://arxiv.org/abs/1801.06146)
- [Understanding AWD-LSTM](https://yashuseth.blog/2018/09/12/awd-lstm-explanation-understanding-language-model/)
- [ULMFiT State of the Art in Text Analysis](https://humboldt-wi.github.io/blog/research/information_systems_1819/group4_ulmfit/)
- [Introduction to Backpropagation Through Time](https://machinelearningmastery.com/gentle-introduction-backpropagation-time/)
