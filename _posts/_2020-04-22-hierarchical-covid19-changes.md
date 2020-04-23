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

Random effects model

$$
\begin{equation*}
  \begin{split}
    y = wt + b + \epsilon
  \end{split}
  \text{, } \qquad \qquad
  \begin{split}
    \epsilon \sim N(0, \sigma^2) \\[10pt]
    p(y \mid w, b, \sigma) \sim N(wt, \sigma^2)
  \end{split}
  \\[15pt]
\end{equation*}
$$

$$
\begin{equation*}
\begin{split} \text{Where: } \qquad \qquad \end{split}
\begin{split}
w &= \begin{cases}
  w_1 & \text{if } \tau \le t\\
  w_2 & \text{if } \tau \gt t\\
\end{cases} \\
b &= \begin{cases}
  b_1 & \text{if } \tau \le t\\
  b_2 & \text{if } \tau \gt t\\
\end{cases}
\end{split}
\\[10pt]
\end{equation*}
$$
