---
layout: post
title: "Particle Fitler Intuition"
date: 2022-03-03 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Intuition behind the particle filter
---

## Problem

The likelihood is complicated and intractable in latent variable models because we have to integrate over the entire latent space. So how can we do inference on these models if we don't have a likelihood function? What does the likelihood we're interested in look like? 

## Particle Filters

### Likelihood Function

In our latent variable model setup, $X_t \in \mathbb{R}^D$ is latent, and $y_t \in \mathbb{R}^D$ is observed.

- Transition density: $p(X_t \mid X_{t-1}, \theta)$. For stochastic differential equations, this is given by the Euler-Maruyama approximation for $\Delta t$. Which we can make more accurate by increasing the resolution number.
- Measurement model: $p(y_t \mid X_t, \theta)$
- Prior on initial state: $\pi(X_0)$

We are interested in the likelihood of the observed data, $y_{0:T}$ under our latent variable model. 

$$ 
\begin{aligned}
L(\theta \mid Y_{0:T}) &= \prod_{t = 0}^T p(y_t \mid \theta)  \\
\end{aligned}
$$

However, we don't have the density $p(y_t \mid \theta)$, and $y_t$ is only related to $\theta$ through $X_t$. This is where the difficulty begins: 

$$ 
\begin{aligned}
p(y_t \mid \theta) = \int_X p(y_t \mid X_t) p(X_t \mid X_{t-1}, \theta) dX_t \qquad \text{DOUBLE CHECK ME PLZ}
\end{aligned}
$$
<!-- 
$$
\begin{aligned}
X_0 \sim p(x_0) & \qquad \qquad \text{Initial state} \\
p(X_t \mid X_{t-1}) &  \qquad \qquad \text{Transition density} \\
p(Y_t \mid X_t) & \qquad \qquad \text{Marginal of $Y_t \mid X_t$?}
\end{aligned}
$$ -->

The final likelihood we are interested in looks like: 

$$ L(\theta \mid Y_{0:T}) = \int \pi(X_0) \prod_{t = 0}^T p(y_t \mid X_t, \theta) \prod_{t = 1}^T p(X_t \mid X_{t-1}, \theta) dX_{0:T} $$

$$ L(\theta \mid Y_{0:T}) = \int \pi(X_0)p(y_0 \mid X_0, \theta) \prod_{t = 1}^T p(y_t \mid X_t, \theta) p(X_t \mid X_{t-1}, \theta) dX_{0:T} $$

## Particle Filtering

The basic idea of particle filtering is to simulate points from $\pi(X_0)$ and propagate these points using the transition density, reweighting them at each timestep using the measurement model, $p(y_t \mid X_t)$. 

At a given timestep, $t$, the reweighted point cloud of particles approximates the distribution $p(y_t \mid X_{t}, \theta)$. 