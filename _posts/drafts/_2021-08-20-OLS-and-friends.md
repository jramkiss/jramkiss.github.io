---
layout: post
title: "OLS and Friends"
date: 2021-08-02 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Notes and interesting properties of OLS regression.
---



## Coefficient of Determination and Correlation 

We can show that another way to derive the coefficient of determination is from the corelation between the observed, $y_i$ and fitted values, $\hat{y}_i$. 

$$ r^2_{\hat{y}, y} = (\frac{Cov(y, \hat{y})}{\sqrt{Var(y)Var(\hat{y})}})^2 $$

$$ = \frac{Cov(y, \hat{y})Cov(y, \hat{y})}{Var(y)Var(\hat{y})} $$

$$ = \frac{Cov(\hat{y} + \epsilon, \hat{y})Cov(\hat{y} + \epsilon, \hat{y})}{Var(y)Var(\hat{y})} $$

$$ = \frac{Var(\hat{y})^2}{Var(y)Var(\hat{y})} $$

$$ = \frac{Var(\hat{y})}{Var(y)} $$

$$ = \frac{\frac{1}{n} \sum_{i = 1}^n (\hat{y}_i - \bar{\hat{y}})^2}{\frac{1}{n} \sum_{i = 1}^n (y_i - \bar{y})^2} $$

Since $\bar{\hat{y}} = E(\hat{y}) = E(y - \epsilon) = E(y)$, we have:

$$ = \frac{ \sum_{i = 1}^n (\hat{y}_i - \bar{y})^2}{\sum_{i = 1}^n (y_i - \bar{y})^2} $$

Another proof of this can be found [here](https://math.stackexchange.com/questions/129909/correlation-coefficient-and-determination-coefficient/1799567).