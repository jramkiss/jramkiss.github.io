---
layout: post
title: "Detecting Change Points in COVID-19 with Bayesian Models"
date: 2020-04-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Usinng Bayesian change point analysis with regression to determine when COVID-19 cases started to change in different countries.
---

## Problem

There's an amazing example of Bayesian change point analysis in the book [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_TFP.ipynb), and

With the current global pandemic and its associated resources (data, analyses, etc.), I've been trying for some time to come up with an interesting COVID-19 problem to attack with statistics. After looking at the number of confirmed cases for some counties, it was clear to me that at _some_ date, the number of new cases stopped being exponential and its distribution changed. However, this date was different for each country (obviously). I propose a Bayesian model for estimating the date that the number of new confirmed cases in a particular country.


## Model

We want to describe $y$, log of the number of new confirmed cases each day, which we'll do using a segmented regression model. The point at which we segment will be determined by a learned parameter, $\tau$. The model is below:

**Likelihood:**

$$
\begin{equation*}
  \begin{split}
    y = wt + b + \epsilon
  \end{split}
  \text{ , }
  \begin{split}
    \epsilon \sim N(0, \sigma^2) \\
    y \sim N(wt, \sigma^2)
  \end{split}
\end{equation*}
$$

$$
\begin{equation*}
\begin{split} \text{Where: } \end{split}
\begin{split}
w = \begin{cases}
  w_1 & \text{if } \tau \le t\\
  w_2 & \text{if } \tau \gt t\\
\end{cases} \\
b = \begin{cases}
  b_1 & \text{if } \tau \le t\\
  b_2 & \text{if } \tau \gt t\\
\end{cases}
\end{split}
\end{equation*}
$$

**Priors:**

$$
\begin{equation*}
  w_1, w_2 \sim N(\mu_w, \sigma_w^2)
  \\
  b_1, b_2 \sim N(\mu_b, \sigma_b^2)
  \\
  \tau \sim Beta(\alpha, \beta)
  \\
  \sigma \sim U(0, 2)
\end{equation*}
$$


Simply put, in the end we want one regression model 1 to describe the data from day 0 to day $\tau$, and regression model 2 to describe the data otherwise.

## Data

The data we'll be looking at was downloaded from [Kaggle](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset#covid_19_data.csv).
