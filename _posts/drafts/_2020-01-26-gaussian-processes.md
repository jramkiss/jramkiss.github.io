---
layout: post
title: "Ease into Gaussian Processes"
date: 2020-01-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Explanation of Gaussian processes for dummies like myself
---


### Why is this a cool model?

Imagine learning about neural networks for the first time, and thinking _"wow, is there anything this thing can't do??"_. And the heartbreak later on, when you realize a [neural network trained to classify cats and dogs](https://jramkiss.github.io/2020/07/29/overconfident-nn/) predicts you're a dog with 98% confidence. Clearly they aren't _that_ good, since they don't even know what they don't know. This is why I find Gaussian Processes so cool, they are the gold standard for "knowing when you don't know".

&nbsp;

### How to start thinking about Gaussian Processes?

I've found many articles about Gaussian processes that start their explanation by describing stochastic processes, then go on to say that a GP is a distribution over functions, or an infinite dimensional distribution. This may be the "right" way to approach it, but I find it harsh for an introduction. My approach to explaining GP's is much more approachable.

One useful way to start thinking about Gaussian processes is to build up from the relationship between the univariate and multivariate Gaussian distributions. Starting with a single random variable, $X_1 \sim N(\mu_1, \sigma_1^2)$, we can append another random variable $X_2 \sim N(\mu_2, \sigma_2^2)$ to obtain a multivariate Gaussian for $(X_1, X_2)$. If $X_1$ and $X_2$ have covariance $\sigma_{12}$, this distribution will look like: 

$$ \begin{pmatrix} X_1 \\ X_2 \end{pmatrix} \sim N \left(\begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}, \begin{bmatrix} \sigma_1^2 & \sigma_{12} \\ \sigma_{12} & \sigma_2^2 \end{bmatrix} \right)$$

If we continue appending more Normally distributed random variables, $X_3, X_4, ...$ we can continue to construct multivariate Gaussians, all we need is their mean and covariance with all other variables. We can generalize this concept of continuously incorporating Normally distributed RV's into the same distribution, if we have a function that describes the mean and another to describe the covariance. 

This is exactly what the Gaussian process provides. It is specified by a mean function, $\mu(X)$ and a kernel function, $k(X, X')$, that returns the covariance between $X$ and $X'$. Now we can model any amount (possibly infinite) of variables with the GP using the mean and kernel function. Since the GP can model an infinite number of random variables, this is why GP's are considered a distribution over functions, and it is written as: 

$$ f(x) \sim GP(\mu(x), k(x, x'))$$ 

An exmaple of the mean and kernel functions are: 

$$ \mu(x) = 0  $$

$$ k(x, x') = \sigma^2 \exp(-\frac{(x - x')^2}{2l^2}) $$

This is the squared exponential kernel. There are many more kernels and a great writeup on them can be found [here](https://www.cs.toronto.edu/~duvenaud/cookbook/).

&nbsp;


### How to fit a Gaussian Processes?







## A Note on Regression

Let's start by explaining different types of linear regression. In simple linear regression, we first make a linearity assumption about the data (we assume the target variable is a linear combination of the features), then we estimate model parameters based on the data. In Bayesian linear regression, we make the same linearity assumption, however we take it a step further and make an incorporate beliefs about the parameters into the model (priors), then learn the parameters from the data.
Gaussian Process Regression takes a different approach. We don't drop the linearity assumption, and the priors on the parameters. Instead we put a prior on **_all possible models_**. As we observe data, the posterior.

**What is Gaussian Process Regression?** - In Gaussian Process regression, a GP is used as a prior on $f$. This means that the posterior distribution over functions is also a GP. The posterior has to be updated every time we observe new data, because the specification of the posterior depends on observed data. Intuitively, the reason we update the GP is to eleminate all functions that do not pass through the observed data points.

### Notes

- The GP is a prior over functions. It is a prior because we specify that we want smooth functions, and we want our points to be related in a certain way, which we do with the kernel.

### Questions

- What about multivariate gaussian? How is this different from the multivariate gaussian?
- What does it mean to "fit a Gaussian process"? What is actually going on in the background?
- Imagine points on a line. If we divide the line into 5 equal points and each point is Normally distributed, this is what a multivariate gaussian would look like, however if we wanted every single one of the points on the line to be normally distributed, this is what a guassian process would look like.
- Can I make an active learner using a GP and the embeddings from a NN to learn 
