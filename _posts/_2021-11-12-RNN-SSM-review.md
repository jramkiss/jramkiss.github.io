---
layout: post
title: "Lit review for RNN + SMC"
date: 2021-09-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: 
---


## Research Directions

- How can we connect RNN's and particle filters? 
- Can we use RNN's for parameter **inference** in latent variable models?
- 

## Questions for myself

- How do the three papers/fields below differ?
    - Neural ODE's / SDE's by Duvenaud 
    - Black-box VI for SDE's
    - Dynamic VAE's
- How does SMC do parameter inference? (Not latent variable inference)

## Lit Review

### RNN's and Latent Variable Models

- [Dynamical Variational Autoencoders: A Comprehensive Review](https://arxiv.org/pdf/2008.12595.pdf): These guys did my job for me apparently. Extensive literature review of using VAE's for sequential data.
- [A Recurrent Latent Variable Model for Sequential Data](https://arxiv.org/pdf/1506.02216.pdf): Variational RNN. Replace NN in a variational auto-encoder with RNN.
- [Structured Inference Networks for Nonlinear State Space Models](https://arxiv.org/pdf/1609.09869.pdf): Deep Markov Models paper. They use a bi-directional RNN to combine observed states and hidden states



- [Physics-guided Deep Markov Models for Learning Nonlinear Dynamical Systems with Uncertainty](https://arxiv.org/pdf/2110.08607.pdf#page=27&zoom=100,84,613): 
- [Deep generative modeling of sequential data with dynamical variational autoencoders](https://dynamicalvae.github.io/): Conference tutorial on combining state-space models with RNN's.
- [Physics-Informed Multi-LSTM Networks for Metamodeling of Nonlinear Structures](https://arxiv.org/pdf/2002.10253.pdf): Combining knowledge of physical systems into LSTM training archtecture



## Deep Markov Models

- [Deep Kalman Filters](https://arxiv.org/abs/1511.05121)
- [Structured Inference Networks for Nonlinear State Space Models](https://arxiv.org/abs/1609.09869)
- [A Recurrent Latent Variable Model for Sequential Data](https://arxiv.org/pdf/1506.02216.pdf): Seminal work, Bengio and Courville on this. 
- [Variational RNN (blog post)](https://medium.com/@deep_space/variational-recurrent-neural-networks-vrnns-3b836adad399)
- [Learning Stochastic Recurrent Neural Networks](https://arxiv.org/pdf/1411.7610v3.pdf): Also seen this paper come up a couple times