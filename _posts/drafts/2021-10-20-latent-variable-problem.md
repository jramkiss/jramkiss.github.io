---
layout: post
title: "The Latent Variable Problem"
date: 2021-10-20 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Latent variables in inference problems
---

I've always heard professors and researchers complain about latent/nuisance variables but never really understood why. Since I'm beginning to get a clearer picture now, I feel like I should write it down so I don't forget. 


## Problem Setup

This section will be intentionally vague to keep generality, but there are some specifics. For starters, we have a model, $f(x \mid \theta)$, that describes some process. This could be stock returns, biological systems, chemical reactions, etc. We know the functional form of $f(x; \theta)$ and our only concern is to find the unknown parameters, $\theta$.

If $X$ was our observed data, then model specification can stop here and we can use the likelihood function to do inference on $\theta$. However, in this problem setup, our observed data is $Y \sim g(y \mid X, \tau)$. Which can be as simple as observational noise, for example, $y \sim N(x, \tau^2)$, or get more complex. 

So the model is: 

$$ X \sim f(x \mid \theta) $$

$$ Y \sim g(y \mid X, \tau) $$

And we can assume that $\tau$ is known. 

In order to do inference on $\theta$, we need a likelihood function, $L(\theta | Y)$. However, $\theta$ is connected to $Y$ through $X$. We must introduce X, then integrate it out. 

$$ L(\theta \mid Y) \propto f(y \mid \theta) = \int p(Y, X \mid \theta) ~ dX $$

$$ = \int p(Y \mid X) p(X \mid \theta) ~ dX $$

