---
layout: post
title: "Dealing with Overconfidence in Neural Networks: Bayesian Approach"
date: 2020-07-29 12:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: I trained a classifier on images of animals and parsed an image of myself it's 97% confident I'm a dog. This is an exploration of a possible Bayesian fix
---

I trained a multi-class classifier on images of cats, dogs and wild animals and parsed an image of myself, it's 97% confident I'm a dog. The problem isn't that I parsed an inappropriate image, because models in the real world are parsed all sorts of garbage. The problem is that the model is overconfident about an image that is far away from the training data instead of having a more uniform distribution over the classes. What makes this particularly interesting is the inability to post-process model output (setting a threshold on predictions, etc.), which means it needs to be dealt with by the architecture.

In this post I explore a Bayesian method for dealing with overconfident predictions for inputs far away from training data in neural networks. The method is called last layer Laplace approximation (LLLA) and was proposed in [this](https://arxiv.org/abs/2002.10118) paper in ICML 2020.

&nbsp;

### Softmax Classifier

The 3-class classifier was trained on images of cats, dogs and wild animals taken from Kaggle that can be downloaded [here](https://www.kaggle.com/andrewmvd/animal-faces?). The model used was Resnet-18, which yields surprisingly good results on the validation data provided.

<p align="center">
  <img src="/assets/overconfident-NN-training-data.png">
</p>


Now for the fun part, parsing an image of myself to the model. For dramatic effect, I also show an image of a dog that the model hasn't seen. Apparently, I'm more dog than this actual dog.

&nbsp;

<p align="center">
  <img src="/assets/overconfident-NN-softmax-predictions.png">
</p>

&nbsp;


### Possible Solutions

Interestingly, [this paper](https://arxiv.org/pdf/1812.05720.pdf) proposes a explanation and proof for the over-confidence of out-of-distribution examples in ReLU networks.   
Essentially they prove that for a given class $k$, there exists a scaling factor $\alpha > 0$ such that the softmax value of input $\alpha x$ as $\alpha \to \infty$ is equal to 1. This means that there are infinitely many inputs that obtain arbitrarily high confidence in ReLU networks. A bi-product of which is the inability to set softmax thresholds to preserve classifier precision.

There are a couple ways this problem can be attacked, which broadly fall into two categories: 1) building a generative model for the data (VAE, GAN, etc.) and 2) changing the structure of the network to assign lower probabilities for inputs far from the training data. The generative approach seems like overkill, and doesn't really solve the problem with ReLU networks. Instead we'll modifying the network directly by injecting Bayesian-ness into the last layer of the model.


### Injecting Bayesian-ness

Bayesian methods are perfect for quantifying uncertainty, and that's what we want in this case. The problem is that this model (and all other deep learning models) have way too many parameters to have an appropriate posterior over. **The proposed solution is to only have a posterior over the last layer of weights.** This is perfect for implementation because we can in theory have the best of both worlds - first use the ReLU network as a feature extractor, then a Bayesian layer at the end to quantify uncertainty.

How do we actually have a posterior over the last layer weights? 

A great explanation of Laplace approximation can be found [here](http://www2.stat.duke.edu/~st118/sta250/laplace.pdf).

<p align="center">
  <img src="/assets/overconfident-NN-out-of-sample-predictions.png">
</p>


### More Testing

Testing a couple hand picked images wasn't sufficient, I want to make sure that the Laplace approximation wasn't naively scaling down the confidence of predictions, and was actually doing something interesting. Here is a plot of the confidence level of the class predicted using softmax and LLLA

<p align="center">
  <img src="/assets/overconfident-NN-top-class-prob-distribution.png">
</p>

It's obvious that LLLA is doing some interesting scaling to the confidence levels, but we can't stop here! What images are predicted with high probability by the softmax model but low probability by LLLA?


<!--

&nbsp;

### Adversarial Confidence Enhancing Training

> We assume that it is possible to characterize a distribution of data points pout on the input space for which we are sure that they do not belong to the true distribution $p_{in}$ resp. the set of the intersection of their supports has zero or close to zero probability mass.
An example of such an out-distribution $p_{out}$ would be the uniform distribution U(0, 1) on gray scale images or similar noise distributions. Suppose that the in-distribution consists of certain image classes like handwritten digits, then the probability mass of all images of handwritten digits under the $p_{out}$ is zero (if it is really a low-dimensional manifold) or close to zero.

The proposed solution is to adjust the loss function to enforce low confidence in the neighborhood of all out-of-distribution points. This way, we implicitly learn a distribution for each class, $p_{in}$ and an out-distribution, $p_{out}$.
The new loss function is below:

$$
\frac{1}{N} \sum^N_{i=1} L_{CE}(y_i, f(x_i)) + \lambda E[\max_{||u - z|| \le \epsilon} L_{p_{out}}(f, u)]
$$

Where $L_{CE}$ is the cross entropy loss (what we would use as the original loss function) and $L_{p_{out}}$ is the max log confidence over all classes.

$$
L_{p_{out}} = \max_{l = 1..K} \log(\frac{e^{f_l(x)}}{\sum^N_{i=1} e^{f_l(x)}})
$$

This loss function makes sense. Consider 2 inputs, $x_{music}$ and $x_{not\ music}$ that are both predicted as sheet music by the model, $f$. The loss at

-->
