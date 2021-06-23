---
layout: post
title: "Why are Bayesian models better at uncertainty estimation?"
date: 2021-01-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: In this post I investigate why Bayesian models are the gold standard for uncertainty estimation and why they are so much "better" than frequentist methods in this regard
---


## Key Questions
- Is the overconfidence problem in neural networks the same as uncertainty quantification? Are these the same as well calibrated models?



## Annotated Bib

These papers may have some interesting and relevant material in them:
- [Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks](https://arxiv.org/abs/2002.10118) - Related work, section 3, talks about beliefs about Bayesian models fixing the overconfidence problem. Also validates Bayesian GLM's and Bayesian last-layer methods.
- [Being a Bit Frequentist Improves Bayesian Neural Networks](https://arxiv.org/abs/2106.10065)
- [An Infinite-Feature Extension for Bayesian ReLU Nets That Fixes Their Asymptotic Overconfidence](https://arxiv.org/abs/2010.02709)
- [Fast Predictive Uncertainty for Classification with Bayesian Deep Networks](https://arxiv.org/abs/2003.01227)


Other papers: 

- [](https://authors.library.caltech.edu/13796/1/MACnc92d.pdf): Apparently this paper shows that Bayesian methods deal with overconfidence.
- [Evidential Deep Learning to Quantify Classification
Uncertainty](https://arxiv.org/pdf/1806.01768.pdf): Authors use a Dirichlet distribution to model NN output 
- [Bayesian DL and Uncertainty Thesis](https://uwspace.uwaterloo.ca/bitstream/handle/10012/15056/Phan_Buu.pdf?isAllowed=y&sequence=3): Introduces key differences between Bayesian and frequentist methods. 
- [Frequentism and Bayesianism: A Python-driven
Primer](https://arxiv.org/pdf/1411.5018.pdf): Nice examples about frequentist VS Bayesian inference
- [Confidence Intervals VS Bayesian Intervals](https://bayes.wustl.edu/etj/articles/confidence.pdf): E.T Jaynes good work
- [Another thesis](https://people.csail.mit.edu/lrchai/files/Chai_thesis.pdf): Haven't read but looks decent