---
layout: post
title: "Post Ideas"
date: 2019-11-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Bullet points of ideas for future posts.
---


### Inferring Clusters with Dirichlet Processes
- Implement a dirichlet process clustering model with MNIST
- Explain what Dirichlet processes are
- Can use MNIST with 10 classes, and less than 10 classes and see how it works

### Interesting Relationships between Probability distributions
- Binomial and beta, multinomial and whatever is its corresponding continuous distribution 


### Online Learning with Bayesian Methods
- Read this paper: https://www.ki.tu-berlin.de/fileadmin/fg135/publikationen/opper/Op98b.pdf
- Implement a simple model with online learning and compare the time and accuracy to a model where we re-estimate using the entire dataset everytime we get new data.
- Compare the two models for accuracy, time to predictions and ease of implementations
- Do we have to know about the posterior to do online learning?

### What in the World is a Dirichlet Process
- Basically a small explanation about conjugate priors and the relationship between binonial distributions and beta distributions, then multinomial distributions and dirichlet distributions.
- From there, explain dirichlet processes

### Why Gaussian Process Regression?
- Explain what fitting a GP is
- What is Guassian Process regression and what is the difference between that, regular regression and bayesian regression

### Bayesian Optimization
- Why use Bayesian Optimization
- Can we use bayesian optimization with machine learning? We can start with a ulmfit model and use Bayesian optimization to tune hyperparameters like BPTT, LR, etc.

### Tradeoffs Between Priors and Data
- How do strong priors affect Bayesian models? Is my model going to be subjective if I start with a strong prior that is wrong? Why would I use Bayesian methods if I have no prior beliefs? (all flat priors)
- Do we update the prior during bayesian inference? Or we only update the posterior?
- Walkthrough about the use of priors, when they get used and the effect they have.
    - Are the updated when we update data?
    - If we start with a strong prior that is wrong, how will this affec the model?
    - How long does it take for data to overcome a prior?
- Have a toy example with bayesian linear regression, maybe an example with interpretabble coefficients, like weight VS height. So we can start with a strong prior and not a lot of data, then test out weak priors with lots of data, etc.
