---
layout: post
title: "Bayesian Optimization"
date: 2019-11-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: A tutorial and walkthrough of Bayesian optimization.
---

## Bayesian Optimization

- Start with a function $f(x)$, which we want to optimize (find a global minima / maxima). We want to find $x$ such that $f(x)$ is at its lowest point.
- Observe noisey data from $f$. Call this data point $(x_{init}, y_{init})$
- Specify Gaussian Process kernel (**what are kernels?**) and hyperparameters (number of iterations, noise level, **what else?**)
- Start loop:
    -  Update GP with all existing smaples. In the beginning this will only be the 1 data point.
    - Propose location using acquisition function (Expected Improvement), all data so far, and the current GP.




## Questions
- **What does it mean to fit a GP?** - In Gaussian Process regression, a GP is used as a prior on $f$. This means that the posterior distribution over functions is also a GP. The posterior has to be updated every time we observe new data, because the specification of the posterior depends on observed data. Intuitively, the reason we update the GP is to eleminate all functions that do not pass through the observed data points.

- **What is Gaussian Process Regression?** - Let's start by explaining different types of linear regression. In simple linear regression, we first make a linearity assumption about the data (we assume the target variable is a linear combination of the features), then we estimate model parameters based on the data. In Bayesian linear regression, we make the same linearity assumption, however we take it a step further and make an incorporate beliefs about the parameters into the model (priors), then learn the parameters from the data. Gaussian Process Regression takes a different approach. We don't drop the linearity assumption, and the priors on the parameters. Instead we put a prior on **_all possible models_**. As we observe data, the posterior
9

- What is actually happening when we minimize the acquisition function?

- Do we intelligently propose new locations? Or is it just random?
