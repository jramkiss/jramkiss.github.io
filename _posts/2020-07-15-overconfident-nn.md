---
layout: post
title: "Dealing with Overconfidence in Neural Networks: Bayesian Approach"
date: 2020-07-29 12:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: I trained a classifier on images of animals and gave it an image of myself, it's 97% confident I'm a dog. This is an exploration of a possible Bayesian fix
---

I trained a multi-class classifier on images of cats, dogs and wild animals and parsed an image of myself, it's 97% confident I'm a dog. The problem isn't that I parsed an inappropriate image, because models in the real world are parsed all sorts of garbage. The problem is that the model is overconfident about an image that is far away from the training data instead of having a more uniform distribution over the classes. This makes it difficult to post-process model output (setting a threshold on predictions, etc.), which means it needs to be dealt with by the architecture.

In this post I explore a Bayesian method for dealing with overconfident predictions for inputs far away from training data in neural networks. The method is called last layer Laplace approximation (LLLA) and was proposed in [this](https://arxiv.org/abs/2002.10118) paper published in ICML 2020.

&nbsp;

### Softmax Classifier

The 3-class classifier was trained on images of cats, dogs and wild animals taken from Kaggle that can be downloaded [here](https://www.kaggle.com/andrewmvd/animal-faces?). The model used was Resnet-18, which yields surprisingly good results on the validation data provided.


<p align="center">
  <img src="/assets/overconfident-NN-training-data.png">
</p>

&nbsp;

Now for the fun part, parsing an image of myself to the model. I also show an image of a dog that the model hasn't seen, and apparently I'm more dog than this actual dog.
Ideally, predictions for the image of myself should be close to uniform over all classes, not concentrated in one class.

&nbsp;

<p align="center">
  <img src="/assets/overconfident-NN-softmax-predictions.png">
</p>

&nbsp;


### Possible Solutions

[This paper](https://arxiv.org/pdf/1812.05720.pdf) proposes a nice explanation and proof for the over-confidence of out-of-distribution examples in ReLU networks. Essentially they prove that for a given class $k$, there exists a scaling factor $\alpha > 0$ such that the softmax value of input $\alpha x$ as $\alpha \to \infty$ is equal to 1. This means that there are infinitely many inputs that obtain arbitrarily high confidence in ReLU networks. A bi-product of which is the inability to set softmax thresholds to preserve classifier precision.

There are a couple ways so approach this, which broadly fall into two categories: 1) building a generative model for the data (VAE, GAN, etc.) and 2) changing the structure of the network. The generative approach seems like overkill, and doesn't really solve the problem with ReLU networks. There's a great [Chicken-MNIST](https://emiliendupont.github.io/2018/03/14/mnist-chicken/) blog post that discusses a solution using VAEs. We'll opt for modifying the network by injecting Bayesian-ness into the last layer. Another approach suggested [here](https://arxiv.org/pdf/1812.05720.pdf) changes the loss function, but that's not what we're going to do here.


&nbsp;


### Last Layer Bayesian-ness

Bayesian methods are perfect for quantifying uncertainty, and that's what we want in this case. The problem is that this model, and all other deep learning models, have way too many parameters to have an appropriate posterior over all. **The proposed solution is to only have a posterior over the weights in the last layer.** This is perfect for implementation because we can in theory have the best of both worlds - first use the ReLU network as a feature extractor, then a Bayesian layer at the end to quantify uncertainty.  
The posterior over the last layer weights can be approximated with a [Laplace approximation](http://www2.stat.duke.edu/~st118/sta250/laplace.pdf) and can be easily obtained from the trained model with Pytorch autograd.

Amazingly, the only parameter we have to focus on is $\sigma^2_0$, the variance of the prior on the weights. It governs how conservative the predictions are. As $\sigma^2_0$ increases, the confidence of out-of-distribution predictions decreases, which is what we want. However we cannot naively increase $\sigma^2_0$ as making it too large would cause predictions for images close to the training data to be uniform as well. Decreasing $\sigma^2_0$ causes the predictions to be more and more similar to the softmax predictions. We want a balance between the two extremes.

Now we can use the last layer Laplace approximation to see if it helps the overconfidence issue. Below I ran the same images of myself and the dog through both the model using softmax and last layer Laplace. I'm still a dog, but with much lower confidence, allowing us to potentially set a threshold on the output.

&nbsp;

<p align="center">
  <img src="/assets/overconfident-NN-out-of-sample-predictions.png">
</p>

&nbsp;


### More Testing

So far we've only tested the method with two hand selected images. I want to see if this method just scales down all confident predictions, or if it is doing some interesting stuff under the hood. To start more evaluation, below is a plot of the confidence level for the top predicted class from both models.


<p align="center">
  <img src="/assets/overconfident-NN-top-class-prob-distribution.png">
</p>

&nbsp;

The softmax model is really confident about nearly all the images in the validation set, and LLLA is doing some interesting things to the confidence level. Can't stop now! When does the LLLA model produce high or low confidence predictions?


&nbsp;
<p align="center">
  <img src="/assets/overconfident-NN-LLLA-high-conf.png">
</p>


<p align="center">
  <img src="/assets/overconfident-NN-LLLA-low-conf.png">
</p>

&nbsp;


### Simpsons + Animals

Ok, last thing - what's the confidence level distribution for images that are completely different. This should give us a proxy for how both methods deal with complete garbage thrown at them.  

&nbsp;
<p align="center">
  <img src="/assets/overconfident-NN-simpsons-data.png">
</p>
&nbsp;

I parsed 250 images of Simpsons character faces into the classifier and plotted the confidence level of the top class for both LLLA and the softmax models. Again, we'd expect this distribution to be closer to 0.33 (random chance).

&nbsp;

<p align="center">
  <img src="/assets/overconfident-NN-top-class-prob-out-out-distribution.png">
</p>

&nbsp;

This is a lot better than just using softmax. Of course its usage will depend on the problem and allowable tradeoff between precision and recall for each class, however these results are promising.




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
