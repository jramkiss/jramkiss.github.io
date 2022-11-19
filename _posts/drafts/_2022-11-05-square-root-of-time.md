---
layout: post
title: "Square Root of Time Rule"
date: 2022-11-05 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Proof of the square root of time rule for comparing volatility estimates
---


In finance, we can use the standard deviation of returns to approximate an asset's volatility. However, when we observe data at different frequencies and calculate the standard deviation, higher frequency data (i.e. finely discretized) has lower standard deviation than lower frequency data. [This post](<https://gregorygundersen.com/blog/2022/05/24/square-root-of-time-rule/>) has a great visual example of this using daily and weekly observations of AAPL stock. 

To account for the difference in standard deviation from different levels of aggregation, we scale the statndard deviation estimates. In this post I provide a quick proof as to why this scaling factor makes sense.


- $T$: total timepoints (observations) in finest discretized series. I call this the base series
- $m$: level of aggregation of base series. For example, $m=7$ when we aggregate daily data into weekly data. This new series will have $\frac{T}{m}$ observations
- $X_1, ..., X_T$: log-returns of base series

Note that the goal is to measure the volatility in the underlying asset. We can define a new variable: 

$$
Y_t = \sum_{i=1}^T X_i \\
Var(Y_t) = \sum_{i=1}^T Var(X_i) = T \sigma_X^2 \\
$$

### Standard Deviation of Base Series

When we find the standard deviation of $X_1, ..., X_T$, we approximate $\sigma_X^2$. 


### Standard Deviation of Aggregated Series

