---
layout: post
title: "Dealing with Overconfidence in Neural Networks: Bayesian Approach"
date: 2020-07-15 12:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: This post is on my experience dealing with the overconfidence problem in ReLU networks by using a last layer Laplace approximation
---

This post talks about one way of dealing with wrong and overconfident predictions in neural networks. What makes this problem intriguing is the inability to post-process model output (setting a threshold on predictions, etc.), which means it needs to be dealt with by the architecture.

To demonstrate the problem, we'll use transfer learning to train a multi-class classifier with 3 classes: cat, dog and wild. Then we'll feed the classifier images that are not animals and see how it performs. The dataset used for this is taken from Kaggle and can be downloaded [here](https://www.kaggle.com/andrewmvd/animal-faces?)

It will follow the results in [this paper](https://proceedings.icml.cc/static/paper_files/icml/2020/780-Paper.pdf) closely and can act as a practical summary.

For future, there is also [this paper](https://arxiv.org/pdf/1812.05720.pdf) that tries to solve the same problem with a different approach.

&nbsp;

<p align="center">
  <img src="/assets/overconfident-NN-out-of-sample-predictions.png">
</p>

&nbsp;

Well, turns out these images are both predicted as sheet music with over 95% confidence. Of course an easy and reactive fix to this would be to add images similar to one on the left to the training data and retrain the model. However this is a duct-tape patch and doesn't solve the underlying problem, which is that the model doesn't really learn about sheet music.
Ideally, an image not seen by the training data would be predicted with lower confidence.

&nbsp;

### Formal Problem and Possible Approaches

Interestingly, [this paper](https://arxiv.org/pdf/1812.05720.pdf) proposes a explanation and proof for the over-confidence of out-of-distribution examples in ReLU networks.   
Essentially they prove that for a given class $k$, there exists a scaling factor $\alpha > 0$ such that the softmax value of input $\alpha x$ as $\alpha \to \infty$ is equal to 1. This means that there are infinitely many inputs that obtain arbitrarily high confidence in ReLU networks. A bi-product of this is the inability to set softmax thresholds to preserve classifier precision.


There are a couple ways this problem can be attacked, which generally fall into two categories: 1) building a generative model for the data and 2) changing the structure of the network to assign lower probabilities for inputs far from the training data. The generative approach seems like overkill, and technically also doesn't solve the problem with ReLU networks. Instead we'll focus on modifying the network directly by injecting Bayesian-ness into the last layer of the model.


### A bit Bayesian?

As described in the paper, an alternative to having a posterior over all parameters of the model (which is infeasible given there are millions of parameters), is to only be Bayesian in the last layer of the network. The approach is similar to transfer learning, where we used a pre-trained network to extract features, then train a custom model on the features. Here this custom model is Bayesian

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
