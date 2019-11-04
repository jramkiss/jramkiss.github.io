---
layout: post
title: "One-Page Wonder: ULMFiT"
date: 2019-11-05 19:22
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


# Presentation for Scribd

## Overview of Transfer Learning

- What is transfer learning? - The idea behind transfer learning is that we can train a model to learn general relationships in data, then use this "knowledge" on other tasks.
- Why use transfer learning? - Utilize the power of pre-trained models
- Problems that arise in transfer learning? -

# ULMFiT

- What is ULMFiT? -
  - Intro to AWD-LSTM model

### Training a ULMFiT Model
##### 1) General Domain language model pre-Training
LM is trained on a large general corpus, which is WikiText103 in this case. Here we want the model to learn general features of language like the structure of sentences, subject-verb-object. (Similar to resnet learning about images)

##### 2) Target task language model fine tuning
Now we want to use the knowledge from the general language, but our target dataset is probably from a different distribution, so we fine tune the language model with our target data. This fine tuning happens in the same way as the pre-training, where the model tries to predict the next word in the sentence. This will allow the model to get exposure to the domain-specific language.

##### 3) Target task classifier
Finally we append a classification layer to the LM.


### Under the hood of ULMFiT

#### General-Domain Language Model Pre-Training
In fastai's ULMFiT, an AWD-LSTM model is used as the architecture for the language model. This is similar to a regular LSTM, but with several regularization and optimization techniques added to it. They chose an embedding size of 400.

- Why use ULMFiT over other approaches?

## ULMFiT for Emotion Classification
