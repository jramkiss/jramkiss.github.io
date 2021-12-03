---
layout: post
title: "Inference on SDE's"
date: 2021-09-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: In this post I want to explore stochastic differential equations and why inference on them is difficult 
---


A stochastic differential equation describes the change in some variable, $X$, at a given time, $t$. 

$$ d X_t = \mu(X_{t-1}, \theta) dt + \sigma(X_{t-1}, \theta) dW_t $$

## Why is inference hard?

* The likelihood is intractable: We cannot write down the likelihood. This makes maximum likehood inference nearly impossible, and 
