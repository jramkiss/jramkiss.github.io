---
layout: post
title: "Normalizing Flows"
date: 2021-07-05 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Introduction to normalizing flows
---


## Change of Variables Intuition 

Consider 2 independent probability distributions, $X \sim \text{Normal}(0, 1)$ and $Y \sim \text{LogNormal}(\mu, \sigma)$ and a situation where we only have samples of $X$ and want to get samples of $Y$. For this particular case, it turns out that $Y = f(X) = e^X$. This means that we can get samples of $Y$ by simply taking our samples from $X$ and applying $f$. 

Although we can use this scheme to sample from $Y$, we don't have access to the density of $Y$ yet. This is because when we applied $f$, the domain changed. The figure on the left below is the domain of $X$ and on the right is the domain of $Y$. The dotted line highlights a point on the domain of $X$ that was transformed to the domain of $Y$. These two points have very different probabilities, which we need to account for when we transform $X$ to $Y$. 

To put it differently, consider 2 points in the domain of $X$, $(x, x + \delta_x)$, after applying $f$, we get the corresponding points in the domain of $Y$, $(y, y + \delta_y)$. By changing domains, we need the density of the region of $\delta_x$ to be exactly the same as the density of the region of $\delta_y$.

## Reading Material

- [Blog post on NF](http://akosiorek.github.io/ml/2018/04/03/norm_flows.html)
- [Normalizing Flows in Pyro](https://pyro.ai/examples/normalizing_flows_i.html)
- [Great blog post on NF (Lil'og)](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)
- [Introduction and review of NF Methods](https://arxiv.org/abs/1908.09257)
- [Tutorial on Normalizing Flows (1) - Distributions and Determinants](https://blog.evjang.com/2018/01/nf1.html)
- [Normalizing Flows Tutorial (2)](https://blog.evjang.com/2018/01/nf2.html)
- [NF NeurIPS Tutorial](https://www.youtube.com/watch?v=u3vVyFVU_lI)
- [NF Tutorial video](https://www.youtube.com/watch?v=i7LjDvsLWCg)
- [HINT Github](https://github.com/VLL-HD/HINT)


### Conditional Density Estimation

- [Conditional Density Estimation with Bayesian Normalising Flows](https://arxiv.org/pdf/1802.04908.pdf)
- [Gaussian Process Conditional Density Estimation](https://papers.nips.cc/paper/2018/file/6a61d423d02a1c56250dc23ae7ff12f3-Paper.pdf)
- [Approximate Bayesian Computation via Regression Density Estimation](https://arxiv.org/abs/1212.1479)
