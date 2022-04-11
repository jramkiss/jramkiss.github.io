---
layout: post
title: "A Review of Multiple Imputation in Survey Sampling"
date: 2022-03-24 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Term report for Stat 854 - Survey Sampling
---

# TODO

- [x] Decsribe the types of missing data (item and unit non-response)
- [x] Describe the mechanisms for missing data
  - [ ] Explain ignorability in the missing at random mechanism
- [ ] Briefly present single imputation and maybe one method
- [ ] Describe the multiple imputation process
- [ ] Variance estimation for multiple imputation
- [ ] Multiple imputation as a Bayesian problem

# Introduction 

Dealing with missing data is a practical issue that distracts from the goal of many statistical analyses. In the context of survey sampling, the main influencers of our data imputation strategy are the reason for missing data and the type of missing data. This report will introduce some mechanisms for missing data and provide background into the general missing data problem. It will then briefly describe the sinlge and multiple data imputation strategies, and provide detail into multiple data imputation.


# Background

To motivate the missing data problem, this section will make reference to a toy example survey. Suppose interest is in measuring the effect of a new real estate policy on households. The auxiliary variables, $x$ are different characteristics of each household such as household income, number of children, education level, etc. The response variable, $y$, is binary indicating whether or not the household was affected. With this example in mind, we can walk through the different facets of missing data.


## Item and Unit Non-Response

In general, there are two types of non-response for survey data, namely item non-response and unit non-response. Unit non-response refers to missing auxiliary variables. In this situation, a unit was selected to be sampled but its measurement was not recorded. This can happen for different reasons including refusal of the unit to participate or inaccessibility of the unit during the data collection phase. On the other hand, item non-response refers to missing values in the response variable, $y$. In the toy example, item non-response can happen if a sampled household refuses to answer whether they were affected by the policy. This report will primarily focus on univariate response variables.


## The Missing Data Mechanism

Diving deeper into the missing data problem, we quickly realize that there are many different reasons for missing data. Knowing these reasons will influence our assumptions about the imputation model. In addition, it is common for the auxiliary variables to be known before hand and the response variable be the only source of missingness. This is easy to see in the toy example. For the rest of this report, we will make the assumption that all auxiliary variables are known and the only source of missingness comes from the response, $y$. This section formalizes the reasons for missingness in $y$ into 3 broad buckets, called missing mechanisms.

To make the missing mechansim more concrete, we define the probability of response for unit $i$, conditional on the values of $y_i$ and $x_i$ as: 

$$
\begin{aligned}
\tau_i &= p(\gamma_i = 1 \mid y_i, x_i) \qquad
\gamma_i = \left\{
\begin{array}{ll}
      1 & \text{Unit i responded} \\
      0 & \text{Unit i did not responde} \\
\end{array}
\right.
\end{aligned}
$$


1) Missing Completely at Random (MCAR): In this case, the probability of response for every unit $i$ in the sample is the same, so there is no dependence on the values of $y_i$ and $x_i$. Expressed mathematically, this is: $\tau_i = \tau, \forall i \in S$. When data is missing completely at random, removing them from the analysis does not bias your inference. 

2) Missing at Random (MAR): Most of the time, data is not missing completely at random. In our toy example, it is reasonable to expect that very high income households and very low income households may not have the same probability of response. More generally, _missing at random_ refers to the probability of response being dependent on the values of auxiliary variables, $x$. Note that the dependence is only on the auxiliary variables and not the response variable, $y$. This is a key characteristic of MAR, implying that the missingness is only dependent on the auxiliary variables and not the response variable. 

$$
\begin{aligned}
\tau_i &= p(\gamma_i = 1 \mid y_i, x_i) = p(\gamma_i = 1 \mid x_i)
\end{aligned}
$$


3) Missing not at Random (MNAR): In our toy example, another plausible scenario for ite non-response is that households that are affected by the policy and have low income don't respond. In this case, the probability of response depends on both the auxiliary variables and the response variable, so $\tau_i$ is now a function of $x_i$ and $y_i$. In this case it is important for the missingness to be explicitly modelled, or the resulting inference will be biased. 



### Questions

- What to do? - Can introduce the missing data/data imputation problem, then talk about different methods foro single data imputation for item non-response.
- Final project - Compare different methods for data imputation for item non-response. Do this for binary and continuous responses. Compare bias and MSE, also look at normalized bias (estimated value / true value). If this is over 1.05 the method is unusable. Start with simple methods then scale up 