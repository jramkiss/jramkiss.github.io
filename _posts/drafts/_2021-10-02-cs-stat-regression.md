---
layout: post
title: "Computer scientists and statisticians arrive at linear regression"
date: 2021-09-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Linear regression from 2 perspectives. 
---


The setting for this post is we want and easy way to predict / explain a continuous response, $y$ with $k$ continuous covariates, $X = (x_1, x_2, ..., x_k)$. Well, the easiest way to go about this is to assume that $y$ is a linear function of $X$, which looks something like this: 

$$ y = \beta_0 + \beta_1 x_1 + ... + \beta_k x_k $$

In matrix notation, this is just: 

$$ y = X\beta $$

Now all we have to do is find the values of our $\beta = (\beta_1, ..., \beta_k)$. Here is where things get interesting, because we can go about this in different ways. 

One way to do this is to define a loss function between the predicted and observed values, $J(\beta)$, and minimize this with respect to the parameters. This is a very general and flexible definition. Another way of doing things is to add some additional constraints. First we define the error between the predicted and observed values, which we call $\epsilon$ and require that this error be iid and Normally distributed. Then we continue from here by defining a likelihood function on our data, and maximizing this likelihood to obtain parameter estimates for $\beta$. I'll talk about these two methods and prove both of them.

## Regression with Loss Functions 

For the sake of this post, I'll only look at mean squared error loss. If we use another loss function, our two methods of parameter estimation will have different results.

$$ J(\beta) = \sum_{i = 1}^n (y - X\beta)^2 = (y - X\beta)^T(y - X\beta) $$

The goal now is:

$$ \underset{\beta}{\text{min}} \quad J(\beta) = (y - X\beta)^T(y - X\beta) $$


Since this is quadratic, we can come up with an analytical expression for the minimum by differentiating w.r.t $\beta$ and setting it equal to $0$: 

$$ \frac{\partial}{\partial \beta} J(\beta) = -2X^T(y - X\beta) $$

$$ -2X^T(y - X\beta) = 0 $$

$$ X^TX\beta = X^Ty $$

$$ \beta = (X^TX)^{-1}X^Ty $$

Here's our estimate for $\beta$. We haven't made any assumptions other than $y$ and $X$ are linearly related. So far we haven't talked about probability distributions at all, but that's about to change. 

## Regression with Probability Distributions 

Now we try to derive linear regression using probability distributions. 



## Resources

- https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf
