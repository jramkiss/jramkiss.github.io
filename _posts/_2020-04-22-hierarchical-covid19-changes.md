---
layout: post
title: "Hierarchical Change Point Model for COVID-19 Cases"
date: 2020-04-22 12:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Hierarchical Bayesian model to estimate the date that flattening of new COVID-19 cases started in all countries.
---


# Model

## Random Effects Model - Partial Pooling


$$
\begin{equation*}
  \begin{split}
    y_i \sim N(X_i\beta + \alpha_{j(i)}, \sigma_1^2) \\ \\
    \alpha_j \sim N(W\delta, \sigma_2^2)
  \end{split}
  \text{ } \qquad
  \begin{split}
    \text{For country $i$ in region $j$} \\
  \end{split}
  \\[15pt]
\end{equation*}
$$

$$
\begin{equation*}
\begin{split} \text{Where: } \qquad \qquad \end{split}
\begin{split}
\beta &= \begin{cases}
  \beta_1 & \text{if } \tau \le t\\
  \beta_2 & \text{if } \tau \gt t\\
\end{cases} \\
W &= \begin{cases}
  W_1 & \text{if } \tau \le t\\
  W_2 & \text{if } \tau \gt t\\
\end{cases} \\
\end{split}
\\[10pt]
\end{equation*}
$$

$$\tau_i \sim p(\tau)$$

#### Open Questions:
- This model has an additive random effect. Can we also have a multiplicative random effect?
- Partial pooling should help with having limited amount of data, but we still have to estimate $\tau$ for each country, wouldn't we be back to square 1 with the data problem?
- How would having $\tau_i$ be different for each country impact partial pooling?
