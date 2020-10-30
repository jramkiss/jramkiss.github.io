---
layout: post
title: "Interesting papers from NeurIPS 2020"
date: 2020-10-10 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Notes on some papers I find interesting from NeurIPS 2020.
---

<!-- ### Questions to answer:
- What is the goal of the paper? What problem are they trying to solve
- What assumptions do they make?
- How do they go about doing it?
- No math / very little math. Meant to just get your foot wet. -->

This post briefly outlines some papers I found interesting in NeruIPS 2020. I'm continuously adding to it to as I find time to read more.


### [Uncertainty-aware Self-training for Text Classification with Few Labels](https://arxiv.org/pdf/2006.15315.pdf)

The goal of this paper is to incorporate uncertainty estimates into a self-training framework to improve both sampling of unlabelled data and training of the student model.
The process starts by building a predictive distribution over an unlabelled point, $x_u$. Multiple forward passes are done, each using [MC dropout](https://arxiv.org/pdf/1506.02142.pdf) to get a distribution over $y_u$.
Now we have pseudo-labels for the unlabelled data and need to select which of these will be used for training. This is done by ranking the pseudo-labelled data by the difference in the entropy of $y_u | D_u$ and the expected posterior entropy of $y_u$. Large values indicate the teacher model is confused with $x_u$ and vice-versa. With the rankings, we have a choice of exploring by using difficult data points (high score), or exploiting by using easy data points (low score).
Finally, we have the data to train the student model. However if it is uses naively all uncertainty information is lost. To incorporate this info into the student model, the loss function is altered to penalize wrong classifications of low-uncertainty points more than wrong classifications of high-uncertainty points. This is done by adding the inverse posterior variance $Var(y_u)$ to the student model loss function.


### [Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness](https://arxiv.org/abs/2006.10108)

Here the authors argue that distance awareness is a necessary condition for uncertainty calibration. They outline two conditions distance awareness in the model, which are:

1) Making the last layer distance aware, which can be achieved by using a GP with a shift-invariant kernel.  

2) Making the hidden layers distance preserving so that the distance in the latent space has a meaningful correspondence in the input space.

They propose a method to improve input distance awareness in residual architectures (ResNet, DenseNet, transformers). It replaces the dense final layer by a Gaussian process with an RBF kernel. The posterior variance for **$x^*$** is given by the $L_2$ distance from the training data in the hidden space, and a Laplace approximation is used to approximate the posterior. I'm still not quite sure how this "distance from the training data" is calculated.

Distance is preserved in the hidden space by using spectral normalization. In residual-based architectures, we can regularize the weights of the residual layers, which is proven to preserve distance.


<!-- ### [Energy Based Out-of-Distribution Detection](https://arxiv.org/pdf/2010.03759.pdf) -->

<!--
### [ClusTR: Clustering Training for Robustness](https://arxiv.org/abs/2006.07682)


### [Bayesian Deep Ensembles via the Neural Tangent Kernel](https://arxiv.org/abs/2007.05864) -->
