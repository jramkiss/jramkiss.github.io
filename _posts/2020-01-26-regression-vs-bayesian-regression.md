---
layout: post
title: "Regression VS Bayesian Regression"
date: 2020-01-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: A motivating example on the power of Bayesian regression over simple linear regression.
---

## Problem

We'll take data from a [study](https://diegopuga.org/papers/rugged.pdf) done on the effect of a nation's geography on their GPD.

### Simple Linear Regression
#### Model

$$ y = X\beta + \epsilon $$
$$ \epsilon \sim N(0, \sigma^{2}) $$

### Bayesian Regression

To make the model Bayesian, we have to put priors on the parameters, $\beta$ and $\sigma$.

## Todo
- Write down formulations in a simple way.
- Mention expressiveness of Bayesian model and lack of expressiveness of the frequentist model.
