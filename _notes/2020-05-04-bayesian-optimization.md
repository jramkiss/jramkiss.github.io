---
layout: post
title: "Bayesian Optimization"
date: 2020-05-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Notes on Bayesian Optimization for self reference later on.
---

## Bayesian Optimization

Starting with an outline of the Bayesian Optimization algorithm:

- Start with a function $f(x)$ that we want to maximize
- Place a Gaussian Process prior on $f$
- Evaluate $f$ at an initial point, $x_0$
- Until convergence criteria is fullfilled:
  - Update posterior on $f$
  - Computer a posterior expectation of $f$
  - Sample a new point $x_{new}$ that maximizes some utility of the expectation of $f$.


Now that we have a template, let's fill in the knnowledge gaps by answering these questions:

- What is a Gaussian Process prior and how does it work here?
- What is the "utility" and how do we know where to sample the next point?
- Why do I keep reading about acquisition functions and expected improvement? What are these things?


### Gaussian Process Prior on $f$

In typical Bayesian problems we specify priors for parameters of the model. One way to think about priors is as a weighted domain for parameters. Any parameter value outside the specified prior domain won't be considered in the model, and parameter values with more "weight" will have a higher probability of being considered in the model. In Bayesian Optimization we put a prior over the entire function we want to minimize, $f$.

You read that correctly, we're placing a prior on $f$, which is a function. The Gaussian Process is a generalization of the Gaussian distribution that gives us the flexibility to do this. You can think of a GP as multiple Normally distributed variables in an array which can be of any length, and this length can change, when we obvserve new points for example.

In order to compute the posterior expectation, we need to fit the Gaussian Process to all observed data (realizations of $f$). This will update the posterior on $f$ to now include all of our observations. More on Gaussian Processes can be seen in this post on **Gaussian Process priors**.


<!-- Image showing the posterior before and after we observe 1 datapoint -->


### Acquisition Function - Utility

So far we have only talked about $f$ and not about its parameters, after all we're only after the parameters that minimize $f$. The acquisition function (sometimes called selection function) describes how much utility we get from sampling different values of the parameters. We want to maximize the utility gained from the next sample so we maximize this acquisition function to find where to sample next.

A commom function used as the acquisition function is Expected Improvement.


### NOTES


1)
> Intuitively, it defines the nonnegative expective improvement over the best previously observed objective value, $f_{best}$ at a given location, $x$.

$$
EI(x \mid D) = \int_{f_{best}}^{\inf} (y - f_{best}) p(y \mid x, D) dy
$$

<br/>

2) $\hat{x}$ is the current position of the optimal hyperparemters. We want to maximize the average value of the difference between every possibble $x$ and the current optimal value.
    1) EI is high when the (posterior) expected value of the loss $\mu(x)$ is better than the current best value, $f(\hat{x})$
    2) EI is high when the uncertainty, $\sigma(x)$, around $\hat{x}$ is high

    This is a typical explore / exploit problem that is parameterized by the kernel function. This makes sense, if we maximize EI, we will either sample points that have a higher expected value than $\hat{x}$ or points in regions of $f$ that have not been explored yet. What parameter controls exploring and what parameter controls exploiting?

    $EI(x) = E[max\{0, f(x) - f(\hat{x})\}]$

<br/>

3) > What makes Bayesian optimization different from other procedures is that it constructs a probabilistic model for $f(x)$ and then exploits this model to make decisions about where in $X$ to next evaluate the function, while integrating out uncertainty


## Bayesian Optimization Example

Step through a problem with images at each timestep:

1) Gaussian process prior on $f$
2) First sample(s) of $f$
3) Fit GP prior
4) Acquisition function (with explanation as to why it is 0 at some points)
5) Next iteration




## Pre-Post

Outline of the Bayesian Optimization algorithm:

- Place Gaussian prior on $f$
- Evaluate $f$ at $n_0$ points and set $n = n_0$
- While $n < N$ :
  - Update posterior with all available data
  - Let $x_n$ be a maximizer of the acquizition function over $x$
  - Acquisition function is computed using the current posterior
  - Observe $y_n = f(x_n)$
  - n = n + 1

<br/>

- Using previously evaluated points, $x_{1:n}$, compute a posterior expectation of what the loss of $f$ looks like
- Sample $f$ at a new point $x_{new}$ that maximizes some utility of the expectation of $f$. The utility specifies which regions of the domain of $f$ are optimal to sample from

## Questions

- What is a Gaussian Process prior and how does it work?
- What is this "utility". How do we know where to sample the next point?
- Why is this better than other hyperoptimization methods (grid search, etc)
- If I have prior beliefs about parameters (eg: learning rate), can I encode this in my prior?


## Readings

- https://distill.pub/2020/bayesian-optimization/
- Best paper so far: https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
- Hyperparameter optimization in high dimensional spaces: http://proceedings.mlr.press/v28/bergstra13.pdf
- https://app.sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf
- https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f
- https://github.com/hyperopt/hyperopt
- https://arxiv.org/pdf/1807.02811.pdf
