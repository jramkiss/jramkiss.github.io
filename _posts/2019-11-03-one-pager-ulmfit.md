---
layout: post
title: "One-Page Wonder: ULMFiT"
date: 2019-11-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: All the information you need to know to understand ULMFiT without spending 1 week on it
---

# Overview
To set the scene for the rest of this article, lets assume that the task is to build a classifier that can determine whether poems are happy or sad.


# Stages in Training a ULMFiT Model
## Training Language Model on General Data

This is the most expensive training stage, where we teach a model about the general structure of language eg: a sentence has a subject-verb-object. In ULMFiT, the model architecture used is an AWD-LSTM (think regular LSTM with dropout on steroids) and the data used is from Wikipedia articles. Embeddings for each word in the corpus are learnt in this stage.

**How is this done?**

## Fine-Tuning Language Model on Target Data

Usually in practical NLP tasks, the target data varies slightly from the data the LM was trained on. For this reason, we need to expose the LM to our target data so it can learn its structure and any new words that weren't in the general data. In our example task, the structure of poems is distinctly different from Wikipedia.

#### Variable Length Backpropagation Through Time

Backpropagation through time (BPTT) is the algorithm used to update the weights of an vanilla RNN. It works by calculating and accumulating errors for each item in a sequence, then updating the network weights. This is very computationally expensive, as sequences can be thousands of items long, meaning we have thousands of computations before we update weights. Complex co-adaptions can also be formed. In Variable Length BPTT, each sequence is truncated by a parameter indicating how many tokens to be backpropagated.

## Building the Classifier


# Resources 
