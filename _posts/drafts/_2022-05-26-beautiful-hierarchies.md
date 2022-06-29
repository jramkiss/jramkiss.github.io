---
layout: post
title: "Inside Bayesian Hierarchical Models"
date: 2022-05-26 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: This post dives deeper into Bayesian hierarchical models
---

Hierarchical modeling (also called multi-level modeling) is a powerful class of models that explicitly encodes the structure of data to help inference. They can be interpreted in different ways, which I will try to present in this post. 

TODO:

- Present the Bayesian formulations for no pooling, partial pooling and complete pooling. 
- clearly explain how these differ
- Why does this actually work?
- How does the parameter sharing actually work? 


$$
\tilde{x}_{t}^{i} \\
w_k^{(i)} = \frac{p(y_k \mid \tilde{x}_{t}^{i})}{\sum_{i=1}^n p(y_k \mid \tilde{x}_{t}^{i})} \\
\sum_{t=1}^T \sum_{i=1}^n w_t^{(i)}
$$