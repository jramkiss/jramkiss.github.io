---
layout: post
title: "Gaussian Noise and Mean Squared Error"
date: 2022-01-05 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Parameter estimation in linear regression from 2 perspectives. 
---

In this post I approach parameter estimation in linear regression from two seemingly different perspectives, which end up with the same solution. This may be trivial to most people but I only made the connection recently and find it beautiful. 

In a linear regression setting, we want to explain our response, $y$, using a linear combination of covariates, $X$. The question is how to estimate the parameters, $\beta$, of this linear funciton. The first approach would be to make an assumption about the distribution of errors and use this to form a likelihood function from which we can do maximum likelihood estimation. A second approach could be to completely by-pass any distributional assumption on the error and instead, choose a loss function to minimize. 

At a high level these 2 approaches seem different enough, but as you peel a couple layers back, they're identical. 


## Approach 1: Maximum Likelihood Estimation

In order to do maximum likelihood estimation, we need a statistical model for the data. With the assumption of Normally distributed additive errors, the model of the data becomes:

$$ y_i = X_i\beta + \epsilon_i, \qquad \qquad \epsilon_i \sim N(0, \sigma^2) $$

$$ y_i \sim N(X\beta, \sigma^2) $$

And the likelihood and log-likelihood function follow naturally: 

$$ L(\beta) = \prod_{i = 1}^n f(x_i) $$

$$ L(\beta) = \prod_{i = 1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp{\{\frac{-1}{2\sigma^2}(y - X\beta)^T(y - X\beta)\}} $$

$$ l(\beta) = \sum_{i = 1}^n -\frac{1}{2}\log(2\pi \sigma^2) - \frac{1}{2\sigma^2}(y - X\beta)^T(y - X\beta)$$

To minimize this with respect to $\beta$, set the partial derivative to zero: 

$$ \frac{\partial l}{\partial \beta} = \frac{1}{\sigma^2}X^T(y - X\beta) = 0 $$

$$ \hat{\beta_{\text{MLE}}} = (X^TX)^{-1}X^Ty $$

Of course, this isn't a surprise at all. 


## Approach 2: Mean Squared Error

Now imagine you've never did any statistics and have never heard aobut a distribution, it would be quite difficult to explicitly make a distributional assumption about the errors. However, you still need to find the parameters of the model. To do this, we can minimize the mean squared error between the observed data and the estimated data. 

$$ J(\beta) = \sum_{i = 1}^n (y - X\beta)^2 = (y - X\beta)^T(y - X\beta) $$

The goal now is:

$$ \underset{\beta}{\text{min}} \quad J(\beta) = (y - X\beta)^T(y - X\beta) $$

As usual, set the partial derivative of $J(\beta)$ equal to 0: 

$$ \frac{\partial J(\beta)}{\partial \beta} = -2X^T(y - X\beta) $$

$$ \hat{\beta} = (X^TX)^{-1}X^Ty $$


This is the same estimate from approach 1, but we haven't mentioned any distributions. Given a bit of thought it makes sense, since on the log-scale the Gaussian kernel looks like MSE loss. 
