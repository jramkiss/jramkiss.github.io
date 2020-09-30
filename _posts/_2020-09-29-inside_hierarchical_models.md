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
May need to use a more complex dataset than just simple linear regression. I will need to find a problem where an MCMC struggles to find an appropriate solution with a single prior, but works well with a hierarchical prior.

### Open Questions
- Hierarchical priors make the model more flexible. Why can't we just use a flat prior for added flexibility?
- Specifically in the case of Normal-InverseChiSquared, the resulting posterior distribution is t-distributed, which has heavier tails than just a Normal. How will using a Normal-InverseChiSquared hierarchical prior compare to using a non-hierachical prior with fat tails? What if we just use a t-distributed prior? For this maybe we can try the 8 schools model with a prior with fat tails?
- I want to compare these 3 prior specs on an appropriate problem. Appropriate meaning not too simple:
  - Normal prior
  - Flat prior, to try to induce flexibility without hierarchy
  - Hierarchical prior

The $t_{\nu}(\mu, \sigma^2)$ distribution can be represented with a Gaussian with mean $\mu$ and variance term distributed as an Inverse-$\chi^2$ with $\nu$ degrees of freedom
