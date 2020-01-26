---
layout: post
title: "Post Ideas"
date: 2019-11-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Bullet points of ideas for future posts.
---


### Tradeoffs Between Priors and Data
- How do strong priors affect Bayesian models? Is my model going to be subjective if I start with a strong prior that is wrong? Why would I use Bayesian methods if I have no prior beliefs? (all flat priors)
- Do we update the prior during bayesian inference? Or we only update the posterior?
- Walkthrough about the use of priors, when they get used and the effect they have.
    - Are the updated when we update data?
    - If we start with a strong prior that is wrong, how will this affec the model?
    - How long does it take for data to overcome a prior?
- Have a toy example with bayesian linear regression, maybe an example with interpretabble coefficients, like weight VS height. So we can start with a strong prior and not a lot of data, then test out weak priors with lots of data, etc.


### Why Gaussian Process Regression?
- Explain what fitting a GP is
- What is Guassian Process regression and what is the difference between that, regular regression and bayesian regression

### Bayesian Methods for Determining Changes in Churn Rate
- Post explaining how we can use a bayesian model to see if the churn rate in Mexico changed when we made a decision about entities.
- We can do this by having a hypothesis about dates, then formulating a model with interpretable parameters to see if the parameters match up with date we made the change

### Bayesian Optimization
- Why use Bayesian Optimization
- Can we use bayesian optimization with machine learning? We can start with a ulmfit model and use Bayesian optimization to tune hyperparameters like BPTT, LR, etc.
