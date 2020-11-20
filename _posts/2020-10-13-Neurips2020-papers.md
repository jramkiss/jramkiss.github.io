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


#### [Uncertainty-aware Self-training for Text Classification with Few Labels](https://arxiv.org/pdf/2006.15315.pdf)

The goal of this paper is to incorporate uncertainty estimates into a self-training framework to improve both sampling of unlabelled data and training of the student model.
The process starts by building a predictive distribution over an unlabelled point, $x_u$. Multiple forward passes are done, each using [MC dropout](https://arxiv.org/pdf/1506.02142.pdf) to get a distribution over $y_u$.
Now we have pseudo-labels for the unlabelled data and need to select which of these will be used for training. This is done by ranking the pseudo-labelled data by the difference in the entropy of $y_u | D_u$ and the expected posterior entropy of $y_u$. Large values indicate the teacher model is confused with $x_u$ and vice-versa. With the rankings, we have a choice of exploring by using difficult data points (high score), or exploiting by using easy data points (low score).
Finally, we have the data to train the student model. However if it is uses naively all uncertainty information is lost. To incorporate this info into the student model, the loss function is altered to penalize wrong classifications of low-uncertainty points more than wrong classifications of high-uncertainty points. This is done by adding the inverse posterior variance $Var(y_u)$ to the student model loss function.


#### [Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness](https://arxiv.org/abs/2006.10108)

Here the authors argue that distance awareness is a necessary condition for uncertainty calibration. They outline two conditions distance awareness in the model, which are:

1) Making the last layer distance aware, which can be achieved by using a GP with a shift-invariant kernel.  

2) Making the hidden layers distance preserving so that the distance in the latent space has a meaningful correspondence in the input space.

They propose a method to improve input distance awareness in residual architectures (ResNet, DenseNet, transformers). It replaces the dense final layer by a Gaussian process with an RBF kernel. The posterior variance for **$x^*$** is given by the $L_2$ distance from the training data in the hidden space, and a Laplace approximation is used to approximate the posterior. I'm still not quite sure how this "distance from the training data" is calculated.

Distance is preserved in the hidden space by using spectral normalization. In residual-based architectures, we can regularize the weights of the residual layers, which is proven to preserve distance.


### [Energy Based Out-of-Distribution Detection](https://arxiv.org/pdf/2010.03759.pdf)

To better estimate in-distribution and out-of-distribution examples with no re-training, compare the raw logit scores and not softmax scores. The authors show that the softmax score are not proportional to the likelihood, $p(x)$, because each logit is scaled by the largest logit. I found this one of the most interesting parts of the paper.

They also propose a method for incorporating energy into a training regime, which is done by adding 2 regularization terms to the loss function. The first penalizes the model when it assigns high energy scores to an in-distribution input and the second penalizes the model when it assigns low energy scores to an out-distribution input. 

<!-- When you train this model with in-distribution and out-distribution data, in a way you defeat the purpose of out-distribution, since you explicitly show the model what out-distribution looks like. I wonder if there's a way we can find counter examples to this. So I would train a model with an appropriate in and out distribution dataset, then feed it examples and try to break it to prove that all it does is learn how to discriminate between the data it was given, and not truly learn about the in-distribution data.  -->

<!-- ### [Bayesian Deep Learning and a Probabilistic Perspective of Generalization](https://arxiv.org/abs/2002.08791) -->

<!-- ### [Can I Trust My Fairness Metric? Assessing Fairness with Unlabeled Data and Bayesian Inference](https://arxiv.org/abs/2010.09851) -->

<!-- ### [On the Expressiveness of Approximate Inference in Bayesian Neural Networks](https://arxiv.org/pdf/1909.00719.pdf) -->

<!--
### [ClusTR: Clustering Training for Robustness](https://arxiv.org/abs/2006.07682)


### [Bayesian Deep Ensembles via the Neural Tangent Kernel](https://arxiv.org/abs/2007.05864) -->
