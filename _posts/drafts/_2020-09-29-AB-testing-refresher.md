---
layout: post
title: "Refresher on A/B Testing"
date: 2020-09-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Discusses concepts such as p-values, one tailed and two tailed tests, t-tests, F-tests and in what scenarios to use them.
---

## Concepts

### P-Values
The p-value of a statistical test is used to determine whether or not to reject the null hypothesis. It is the probability of observing the data under the null hypothesis. Smaller p-values simply indicate a lower probability of observing the data, given that the null hypothesis is true. If the p-value is low enough, we can say that the observation is improbable enough for us to reject our belief in the null hypothesis.

The p-value can also be a measure of the false positive rate. Meaning it is the probability of observing the data even though there is no change. This is related to the significance level, which is the threshold at which we reject the null hypothesis.

### Statistical Power
The power of a statistical test is the probability of finding an effect if there is an effect to be found.

### Type 1 and Type 2 Errors

A type 1 error occurs when we wrongly reject the null hypothesis when it is actually true. In plain language: we accept $H_A$ but $H_0$ is true. The probability of this happening will depend on our significance level, $\alpha$, because we automatically reject $H_0$ for $p < \alpha$. Visually, this is the region where distributions for $H_A$ and $H_0$ overlap, so it is also the probability that wrongly reject $H_0$.

A type 2 error occurs when we wrongly fail to reject the null, when the alternative hypothesis is true. In plain language: we accept $H_0$ when $H_A$ is true. The probability of this happening depends on the power of our test, which is the probability of observing an effect given there is one present. This probability is $\beta = p(R | H_A)$. So the probability of wrongly choosing $H_0$ when $H_A$ is true is $1 - \beta$.

More reading [here](https://www.stat.berkeley.edu/~hhuang/STAT141/Lecture-FDR.pdf).

## Statistical Tests

### One Sample t-Tests
A one sample t-test is used to compare a sample mean with a null hypothesis. An example of this problem is: "The average number of site visits per day is 100."

The data collected is summarized by calculating a t-statistic:

$$ t = \frac{\bar{X} - \mu_0}{\frac{s}{\sqrt{n}}} $$

Here we scale the difference between the sample mean, $\bar{X}$ and null hypothesis, $\mu_0$ by the sample standard deviation. Of course as $n$ increases, the denominator becomes smaller, increasing the size of $t$.

### Two-Sample t-Test
A two-sample t-test is used to compare the mean of 2 different populations. It is fundamentally different from the one-sample test because now we have to deal with two sample sizes and a potential difference in population variances. If we suspect the variance of both samples are the same, the test statistic is similar to the one-sample test:

$$ t = \frac{\bar{X_1} - \bar{X_2}}{\frac{s_{pooled}}{\sqrt{n_1 + n_2}}} $$

### Comparing Multiple Variants

In most cases we want to test multiple different variants to see which performs best. To analyze multiple variants naively, one may consider the pairwise comparison of means using t-tests. For 3 variants, A, B and C you may compare A to B, B to C and A to C. However this comparison is inappropriate as it explodes the type 1 error.
To further explain this effect, I'll start by defining the $\alpha$ level, or significance level. This is the probability of wrongly rejecting the null hypothesis. It comes into play when determining the probability of observing the $t$-statistic. If $P(t) < \alpha$, we reject the null hypothesis and vice versa. This means that for large $\alpha$ values we will invariably accept the null hypothesis more than lower $\alpha$ values.
It is important to keep in mind that when we are doing a pairwise comparison, we are looking for a single winning variant. The implication of this is that we need **at least one significant result** out of all repeated tests. Since the probability of falsely rejecting $H_0$ for 1 test is $\alpha$, the probability of obtaining at least one significant result is: $1 - \alpha^k$ where $k$ is the number of times we repeat the test. For 4 variants, $k = 6$. [Here](http://grants.hhp.coe.uh.edu/doconnor/PEP6305/Multiple%20t%20tests.htm) is a good explanation.
The more appropriate way to measure multiple variants is using ANOVA.


### Comparing Multiple Variants with ANOVA

[This](https://statisticsbyjim.com/anova/f-tests-anova/) post describe ANOVA and F-tests.


## Reading Material

- [Stanford course on experimental design](https://statweb.stanford.edu/~owen/courses/363/)
- [Guide to experiments on the web](https://dl.acm.org/doi/10.1145/1281192.1281295)
- [Lanarkshire milk experiment](http://www.medicine.mcgill.ca/epidemiology/hanley/Reprints/RCH/06LanarkShireMilk.pdf)
