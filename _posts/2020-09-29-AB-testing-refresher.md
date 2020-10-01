---
layout: post
title: "Refresher on A/B Testing"
date: 2020-09-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Discusses concepts such as p-values, one tailed and two tailed tests, t-tests, F-tests and in what scenarios to use them.
---


### One Sample t-Tests
A one sample t-test is used to compare a sample mean with a null hypothesis. An example of this problem is: "The average number of site visits per day is 100."

The data collected is summarized by calculating a t-statistic:

$$ t = \frac{\bar{X} - \mu_0}{\frac{s}{\sqrt{n}}} $$

Here we scale the difference between the sample mean, $\bar{X}$ and null hypothesis, $\mu_0$ by the sample standard deviation. Of course as $n$ increases, the denominator becomes smaller, increasing the size of $t$. 
