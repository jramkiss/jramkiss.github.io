---
layout: post
title: "Bayesian Optimization Explained and Applied"
date: 2020-04-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: A look at Bayesian optimization with application to a real world problem
---

## Bayesian Optimization

This post first will focus on explaining Bayesian Optimization, then show an example of its usage in Python. Since you're here, I assume you have some interest in finding out more about Bayesian Optimization, and therefore at least know one of its use cases - hyper parameter optimization.

Let's jump right in by starting with an outline of the Bayesian Optimization algorithm, then we'll disect it.

- Start with a function $f(x)$ that we want to minimize
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

In order to compute the posterior expectation, we need to fit the Gaussian Process to all observed data (realizations of $f$). This will update the posterior on $f$ to now include all of our observations. A visual representation of this can be seen in this post on **Gaussian Process priors**.


<!-- Image showing the posterior before and after we observe 1 datapoint -->




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

- Hyperparameter optimization in high dimensional spaces: http://proceedings.mlr.press/v28/bergstra13.pdf
- https://app.sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf
- https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f
- https://github.com/hyperopt/hyperopt
- https://arxiv.org/pdf/1807.02811.pdf
