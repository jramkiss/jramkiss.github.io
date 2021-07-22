---
layout: post
title: "Normalizing flows for normal people"
date: 2021-07-05 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Normalizing flows for normal people
---


## What (Problem)

Accessible probability distributions such as Normal, Uniform and Exponential provide reasonable explanation for simple parameters, such as regression coefficients. With these distributions, we can better understand parameters. However, if we want to represent more complex structure, like images, we need much more expressive, multi-modal distributions.

One way to construct these complex distributions and have them be reasonably easy to sample from is with a variational auto encoder. However, VAE's have their own share of problems that limit the expressiveness of the resulting distributions. Normalizing flows provide a way to learn expressive distributions that are easy to sample from.

## So What (solution)

Understanding the change of variables formula is probably the easiest way to grasp the concept of normalizing flows. 

### Change of Variables Intuition 

Say we have 2 independent probability distributions, $X \sim \text{Normal}(0, 1)$ and $Y \sim \text{LogNormal}(\mu, \sigma)$ and a situation where we only have samples of $X$ and want to get samples of $Y$. For this particular case, it turns out that $Y = f(X) = e^X$. This means that we can get samples of $Y$ by simply taking our samples from $X$ and applying $f$. 

Although we can use this scheme to sample from $Y$, we don't have access to the density of $Y$ yet, so we can't truly understand how $Y$ is distributed. This is because when we applied $f$, the domain changed. The figure on the left below is the domain of $X$ and on the right is the domain of $Y$. The dotted line highlights a point on the domain of $X$ that was transformed to the domain of $Y$. These two points have very different probabilities, which we need to account for when we transform $X$ to $Y$. 

To put it differently, consider 2 points in the domain of $X$, $(x, x + \delta_x)$, after applying $f$, we get the corresponding points in the domain of $Y$, $(y, y + \delta_y)$. By changing domains, we need the density of the region of $\delta_x$ to be exactly the same as the density of the region of $\delta_y$.


## Now What (benefit)




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
- [ICML 2021 NF Workshop](https://invertibleworkshop.github.io/index.html)
- [](https://arxiv.org/pdf/2007.02731.pdf)


### Conditional Density Estimation

- [Conditional Density Estimation with Bayesian Normalising Flows](https://arxiv.org/pdf/1802.04908.pdf)
- [Gaussian Process Conditional Density Estimation](https://papers.nips.cc/paper/2018/file/6a61d423d02a1c56250dc23ae7ff12f3-Paper.pdf)
- [Approximate Bayesian Computation via Regression Density Estimation](https://arxiv.org/abs/1212.1479)
