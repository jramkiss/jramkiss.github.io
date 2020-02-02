---
layout: post
title: "Regression VS Bayesian Regression"
date: 2020-01-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: A motivating example on the power of Bayesian regression over simple linear regression.
---

## Key Differences
- Bayesian models provide uncertainty estimates, which are important in determining how our model performs (how robust our model is) under certain parameter values.
- Under a Bayesian framework, we can encode knowledge about parameters to supplement the model. For example, consider this toy problem: we are trying to find the error in a piece of apparatus that measures the acceleration of objects. We gather data by measuring dropping objects from a height and measuring their acceleration - which should be close to gravity. This "knowledge" about what the acceleration should be can be encoded into a Bayesian model, but cannot be used in a frequentist model.


## Motivating Problem

To apply both regression methods to a real world problem, we'll try to determine the impact of terrain geography on economic growth for nations in Africa and outside of Africa.

This has been studied [here](https://diegopuga.org/papers/rugged.pdf).

### Simple Linear Regression
#### Model

$$ y = X\beta + \epsilon $$
$$ \epsilon \sim N(0, \sigma^{2}) $$

### Bayesian Regression

To make the model Bayesian, we have to put priors on the parameters, $\beta$ and $\sigma$.

## Todo
- Write down formulations in a simple way.
- Mention expressiveness of Bayesian model and lack of expressiveness of the frequentist model.
