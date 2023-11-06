---
layout: post
title: "Multi-Task Learning"
date: 2023-04-21 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: How do deep learning models learn to do multiple things?
---

In this post I explore how people who actually know what they're doing go about multi-task learning. This was motivated by me naively attempting to train a model on multiple tasks and failing successfully. 

## Current Approaches to Multi-Task Learning

Multi-task learning is also referred to as joint learning, learning to learn, learning with auxilliary tasks, and others. According to [[Caruana, 1998]](https://link.springer.com/article/10.1023/A:1007379606734), “Multi-task learning improves generalization by leveraging the domain-specific information contained in the training signals of related tasks". This makes sense intuitively and is how humans are able to learn so effectively. In general there are two approaches to multi-task learning, which are hard and soft parameter sharing. 

### Hard Parameter Sharing

In hard parameter sharing, the base model is used as a feature extractor that is shared by each task. Compared to single-task learning, hard parameter sharing with N tasks reduces the chances of overfitting by $\frac{1}{N}$. This makes sense since the model now has to find a representation appropriate to each of the N tasks.


Hard parameter sharing requires tasks to be very closely related. If tasks require reasoning on multiple levels, or are not closely related, this approach breaks down quickly. I actually found this empirically in some experiments as well, where a model simply would not learn with 2 tasks that are loosely related. This concept is called _negative transfer_, where tasks hurt each other instead of helping. 

### Soft Parameter Sharing

In soft parameter sharing, each task has its own base model, however the parameters for each base model are regularized to be close to each other. 

## Choosing Tasks 

This section will explore how tasks are chosen and how similar they need to be. I am also interested in how models can borrow information from different tasks to make predictions. In some sense this would be a conditional prediction of task $i$ given predictions/labels for the other tasks. 


## Optimization Problems

## Information Flow

Information flow refers to the sharing of information between tasks. This is important to deal with negative transfer, where increased performance on one task decreases performance on another. 


- [MT-DNN](https://arxiv.org/pdf/1901.11504.pdf): Train BERT with 4 different tasks using hard parameter sharing. Each task shares an embedding layer and a task specific layer is learned. Did not go into detail about how this model is trained
- 

## Questions

- Can the multi-task learning problem be seen as a hierarchical model? In hierarchical models we encode structure into the data through the creation of priors. In multi-task learning we do something similar, where we "group" related tasks and define how they should be learnt. Which is either by learning parameters for models built on a shared representation (hard parameter sharing) or by defining a prior-like structure and regularizing parameters for each model (soft parameter sharing). 

<!-- 
$$
\mathbb{E}[V_d(t)] = \text{EWMA}_{n=9} [V_{d}(t)] \\
\mathbb{E}[V_d(t)] = \sum_{i=0}^{9} \lambda^{i}(1-\lambda) V_{d-i}(t) \\
\mathbb{E}[V_d(t)] = \sum_{i=0}^{9} w_i V_{d-i}(t) 
$$ -->