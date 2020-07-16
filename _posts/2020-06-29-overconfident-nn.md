---
layout: post
title: "Wrong and Strong Neural Networks"
date: 2020-06-29 12:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: This post is on the overconfidence problem in neural networks
---

This post will mainly walk through my experience building an open ended image classifier and dealing with the overconfidence problem with ReLU networks. It will follow the results in [this paper](https://arxiv.org/pdf/1812.05720.pdf) closely and can act as a practical summary. 

I've been struggling with a seemingly simple problem. The task is to build an image classifier to determine if an arbitrary image is sheet music or not. Just like you, on the surface I thought this would be an easy and borderline mundane task - how could this possibly not work??

As a reminder of how "easy" this problem is, here are example images of sheet music and not sheet music.

&nbsp;

<p align="center">
  <img src="/assets/maybe-sheet-music.jpg" height="350">
  <img src="/assets/yes-sheet-music.jpg" height="350">
</p>

&nbsp;

Well, turns out these images are both predicted as sheet music with over 95% confidence. Of course an easy and reactive fix to this would be to add images similar to one on the left to the training data and retrain the model. However this is a duct-tape patch and doesn't solve the underlying problem, which is that the model doesn't really learn about sheet music.
Ideally, an image not seen by the training data would be predicted with lower confidence.

&nbsp;

### Formal Problem and Possible Approaches

Interestingly, [this paper](https://arxiv.org/pdf/1812.05720.pdf) proposes a explanation and proof for the over-confidence of out-of-distribution examples in ReLU networks.   
Essentially they prove that there exists a scaling factor $\alpha > 0$ such that the softmax value of $\alpha x$ for class $k$ as $\alpha \to \infty$ is equal to 1. This means that there are infinitely many inputs that obtain arbitrarily high confidence in ReLU networks. A bi-product of this is the inability to set softmax thresholds to preserve classifier precision.


There are a couple ways this problem can be attacked, which generally fall into two categories: 1) building a generative model for the data and 2) changing the structure of the network to assign lower probabilities for inputs far from the training data. The generative approach seems like overkill, and technically also doesn't solve the problem with ReLU networks. Instead we'll focus on modifying the network directly, as proposed by [this paper](https://arxiv.org/pdf/1812.05720.pdf).
