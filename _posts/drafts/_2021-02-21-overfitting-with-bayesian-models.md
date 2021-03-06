---
layout: post
title: "Overfitting a Bayesian Regression"
date: 2021-02-22 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Do Bayesian models overfit? One way of overfitting a model is training for too long. How does this transfer to Bayesian models with no notion of "training"?
---

When training a deep neural network, or any model using gradient descent, it's easy to be weary of overfitting by not running the optimization for too many steps (over optimizing on training set). I've been thinking of the corresponding problem for a Bayesian regression, where there's no notion of optimization steps, only samples drawn from a posterior. Surely just this doesn't alleviate overfitting. Also, can Bayesian models even overfit? I don't (didn't) know the answers. 


## Regularization with Maximum Likelihood and MAP 

To start, I took a step back and thought about how we would deal with overfitting in another model that doesn't need optimization from gradient descent - simple linear regression. Of course, the easiest thing to do here is to add a penalty to the loss function, and if we still want a closed form solution we add the L2 norm. Parameter estimation from here is relatively straightforward, and in their [respective solutions](https://math.stackexchange.com/questions/2917109/map-solution-for-linear-regression-what-is-a-gaussian-prior), its easy to see the influence of . 




## Regularization with Bayesian Inference

For some reason I always thought of priors as these mysterious things that should be as flat as possible unless you are reasonably confident about some parameters. That way you induce as little subjectivity into the model as possible. However I realize now that this isn't always the case. 

Adding L1 and L2 penalties to the loss function in MLE regression simply restricts parameters from becoming too large. This can be accomplished in a Bayesian setting by lowering the variance of your priors. So what can be referred to as a "strong prior", doesn't always have to reflect the amount of prior knowledge we have about a parameter, it can also be interpreted as the amount of regularization applied to a particular parameter. 

## Overfitting a Bayesian Regression Example

Here we'll look at 2 very similar models, one with flat priors and another with strong priors and test their ability to generalize to unseen data. This ability to generalize will be a proxy for how much the model is overfit to the training data. 

