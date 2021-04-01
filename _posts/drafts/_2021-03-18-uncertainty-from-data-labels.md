---
layout: post
title: "Overconfidence in Data Labels"
date: 2020-10-10 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Thoughts on how data labelling impacts uncertainty quantification in neural networks
---

When labelling data for multiclass problems, we generally have $x_i \in \{0, 1\}$ and $\sum_{i} x_i = 1$. So each datapoint is forced to be in a single class. We then explicitly teach the model that points further away from the class centroid have the same "score" as points closer to the class centroid, which is incorrect. 

From a different perspective, say we are training a softmax neural network to classify MNIST digits. The target variable for each digit is a one-hot encoded vector of length 10. We calculate the loss between predictions and these one-hot encoded arrays, and expect the networks to implicitly learn uncertainties from the data. However, the target variable itself is misspecified. Because the target variables are one-hot encoded, we teach the network that there is no difference between a well written $3$ and a $3$ that looks like an $8$. 

With this in mind, it seems as though neural networks infer epistemic (PLZ CONFIRM) uncertainty from the training data, and don't explicitly learn it. This seems fragile and very sensitive to variation in trainng data distributions. 


## Proposed Solution

One proposed solution is an active learning / self-supervised approach, where we iteratively train models on increasingly larger datasets, but with a key difference in the training scheme. Instead of one-hot encoding model predictions, we use the raw model output as training data for the next cycle. This way we are able to teach the model subtle differences between classes, if overlap exists. For example, a $3$ that looks like an $8$. We ideally want the model to be confused about whether or not this is a $3$ or an $8$, and not overconfident in either one. 


## Resources

- [Aleatoric and Epistemic Uncertainty in Machine Learning: An Introduction to Concepts and Methods](https://arxiv.org/pdf/1910.09457.pdf)
- 