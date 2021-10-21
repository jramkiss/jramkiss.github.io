---
layout: post
title: "ML Interview Questions and Answers"
date: 2021-09-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: 
---

All questions from Chip Huyen's [book on ML interviews](https://huyenchip.com/ml-interviews-book/contents/5.1.1-vectors.html). [Here](https://huyenchip.com/machine-learning-systems-design/toc.html) is the ML system's design book.


## Vectors

1) What is the geometric interpretation of a dot product? 

Geometrically, the dot product between $a$ and $b$ is the length of a multiplied by the length of b, multiplied by the angle between both. 


2) Given a vector , find vector of unit length such that the dot product of and is maximum.


## Probability 

2) Can the values of PDF be greater than 1? If so, how do we interpret PDF?

Yes, the values of a PDF can be greater than 1. The value of the PDF at a certain point isn't the probability observing that point. To go from PDF to probability, we have to integrate across a range. For a random variable $X \sim U(0, \frac{1}{2})$, the PDF is 2. Even so, the area under the curve (total probability) is still 1. 

3) What’s the difference between multivariate distribution and multimodal distribution?

A multivariate distribution has multiple variables possibly interacting and influencing each other. For example, we can think of modelling a human as a multivariate distribution where two of the variates are hieght and weight. A multimodal distribution however, is a distribution with more than one distinct "peak" or mode. For example, coding ability in an introductory CS class is usually multi-modal.

4) What does it mean for two variables to be independent?

For two variables to be independent, means that any change in one has no impact on the other. 

5) It’s a common practice to assume an unknown variable to be of the normal distribution. Why is that?

Many reasons. One is that the Normal distribution is easy to conceptualize and is symmetric. However there are many symmetric distributions (t-distribution). The Normal distribution usually comes with mathematical or computational ease of use.

6) How would you turn a probabilistic model into a deterministic model?

A probabilistic model has probability distributions for each parameter. During inference time, we integrate over all values of the parameters to form a prediction. To change a probabilistic model into a deterministic model, we can instead use the mean/median/mode of the parameter distributions. 

7) Is it possible to transform non-normal variables into normal variables? How?

Can we use the CLT for this? 

8) When is the t-distribution useful?

Two use cases come to mind: 
- Robust regression: if we notice larger than usual errors in our regression, we can model the errors with a t-distributions rather than a Normal. This is because the t-distribution has fatter tails.
- Hypothesis testing: When testing whether a regression coefficient is equal to 0, the test statistic under the null hypothesis follows a t-distribution

9) Assume you manage an unreliable file storage system that crashed 5 times in the last year, each crash happens independently.
- What's the probability that it will crash in the next month?
- What's the probability that it will crash at any given moment?

Crash rate per month is $\frac{5}{12}$.

$$ X: \text{Number of crashes next month} $$ 

$$ X \sim Poi(\lambda) $$
$$ X \sim Poi(\frac{5}{12}) $$

$$ p(X = 1) = \frac{\frac{5}{12} e^{-\frac{5}{12}}}{1!} $$



## Statistics

14) There’s a rare disease that only 1 in 10000 people get. Scientists have developed a test to diagnose the disease with the false positive rate and false negative rate of 1%.

    - Given a person is diagnosed positive, what’s the probability that this person actually has the disease?
    
    - What’s the probability that a person has the disease if two independent tests both come back positive?

**Solution**:

Part (a):

Part (b):

A: has disease
B: see a negative test

$$ p(A \mid BB) = p(A \mid B)^2 $$

$$ p(B) = p(B \mid A)p(A) + p(B \mid A') p(A') $$

$$ p(B) = FNR * p(A) + [1 - FNR] * p(A') $$

$$ p(B) = \frac{1}{1e6}(1 + 99(9999)) $$

$$ p(A, B) = p(B \mid A)p(A) $$

$$ p(B \mid A)p(A) = \frac{1}{1e6} $$

$$ p(A \mid BB) = p(A \mid B)^2 $$ 

$$ p(A \mid B)^2 = \frac{1}{1e12} * \frac{1 + 99(9999)}{1e6}^2 $$