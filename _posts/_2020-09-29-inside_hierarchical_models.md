---
layout: post
title: "How are hierarchical models so different?"
date: 2020-09-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Using a hierarchical prior changes Bayesian models a lot more than it initially appears. Here I explore why
---

I was wondering what makes hierarchical models so much more flexible than non-hierarchical models, since the concept seems relatively straight foward. Also, why not just use flatter priors if we want a more flexible model? What's this secret hierarchical sauce?
To illustrate the difference between a hierarchical model and a non-hierarchical model I'll use a simple linear regression model. I'll take a dataset with a couple outliers and we'll analyze the posterior of the coefficient term under 3 different priors:

- Normal prior
- Flat prior, to try to induce flexibility without hierarchy
- Hierarchical prior

The $t_{\nu}(\mu, \sigma^2)$ distribution can be represented with a Gaussian with mean $\mu$ and variance term distributed as an Inverse-$\chi^2$ with $\nu$ degrees of freedom
