---
layout: post
title: "Refresher on Importance Sampling"
date: 2022-10-22 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Importance sampling
---


Long story short, importance sampling is a method for approximating integrals using samples from a probability distribution. It is widely used in statistics, since many quantities of interest can be expressed as integrals. The idea behind importance sampling is to first express the integral in terms of some probability measure, then change the measure to something easier to work with.

The token example of one such quantity is the expectation of a random quantity $f(X)$ with respect to some probability measure (probability distribution), $P$: 

$$
E_{P}(f(X)) = \int_{-\infty}^{\infty} f(x) p(x) dx
$$

One way to approximate this integral is to use crude Monte Carlo, where we sample $x_1, ..., x_n$ from $P$. Then an estimate of $E_{P}(f(X))$ is: 

$$
\hat{E}_{P}(f(X)) = \frac{1}{n} \sum_{i=1}^n f(x_i)
$$

This estimate seems fine at first glance, but we very quickly run into problems. What happens if we can't sample from $P$? Importance sampling provides a solution to this problem by changing the measure (density) from $P$ to $Q$. 


## Importance Sampling Intuition 

We can re-write an arbitrary expected value of $f(X)$ w.r.t. the probability measure $P$ as: 

$$
\begin{aligned}
E_{P}(f(X)) &= E_{P}(f(X) \frac{q(X)}{q(X)}) \\
                      &= \int_{-\infty}^{\infty} f(x) \frac{q(X)}{q(X)} p(x) dx \\
                      &= \int_{-\infty}^{\infty} f(x) \frac{p(X)}{q(X)} q(x) dx \\
                      &= \int_{-\infty}^{\infty} w(X) q(x) dx
\end{aligned}
$$

Here I let $w(X) = f(x) \frac{p(X)}{q(X)}$ to illustrate that this integral starts to look like $E_{Q}(w(X))$. Remember the problem we had with using crude Monte Carlo on $P$ was that it was too difficult to sample from. However, we have just expressed an expectation over $P$ in terms of another measure, $Q$. If $Q$ is easier to sample from, then we can use crude Monte Carlo and circumvent the original sampling difficulties with $P$. Thereby, approximating an expectation over $P$ without ever sampling from it. 

The story isn't over yet, since we haven't talked about $Q$ at all. We have to impose some conditions on how $Q$ is chosen, and there is both theory and intuition to back this up. For starters, if we look at how $w(X)$ is defined: 

$$
w(X) = f(x) \frac{p(X)}{q(X)}
$$

If $q(x) = 0$ and $p(x) > 0$, our integral is undefined. So we must impose the condition that if $f(x) = 0$, then $q(x) = 0$, $\forall x$. This constraint is called absolute continuity, and is related to the use of the [Radon-Nikodym Theorem](https://mathworld.wolfram.com/Radon-NikodymTheorem.html) for the change of measure. 

To be continued...


<!-- ## When is this useful? 

Importance sampling is particularly useful when we can evaluate $P$ but it is difficult or impossible to sapmle from. These situations are surpringly common, especially in Bayesian statistics. 

Importance sampling is also useful when we are interested in simulated tail events. To see this we can consider trying to evaluate $E_{P}(f(X) \mid X > 5)$, where $X \sim N(0, 1)$. One way of doing this using crude Monte Carlo is to sample $x^* \sim N(0, 1)$ and accept $x^*$ if it is greater than 5, reject otherwise. Then after collecting enough values of $x^*$, we can take the average $\frac{1}{n} \sum_{i=1}^n f(x^*_i)$. It is not hard to see that we will end up rejecting 

## Example: Tail Expectation -->