---
layout: post
title: "The Beauty and Pain of Hierarchical Models"
date: 2023-02-11 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: A single unifying post on hierarchical models
---

# Goal

This post has been a while in the making and every time I see hierarchical structure in the world I want to write it. The ultimate goal of this post is to show different kinds of hierarhical models and how they show up and can be used in different areas of life. I have used them in many different settings so far, so a single unifying post would be great. Here are the goals for this post: 

- Unify diffferent thought processes behind hierarchical models. For example, sometimes hierarchy shows up naturally, such as topclass-subclass relationships. Other times, for whatever reason, hierarchy seems hidden. Present different projects and their hierarchical models to see if I can come to some similarities. Here are some example projects that can be included: 
  1) Taxonomy project at Scribd: The class structure was already defined and I used an interesting structure for the prior distribution to induce hierarchy in the model parameters
  2) RF at DRW: The structure arose naturally from the links, but the modelling approach was very different from Scribd as I did not explicitly encode the structure into the prior
  3) LDA: In order to allow documents to have more than one topic, a hierarchical approach is taken (compared to a 2-level Dirichlet-Multinomial model)

- Why do hierarchical models work so well? What is **really** going on? 
- How to make a model hierarchical from first principals? Do we simply add a level of hyper-priors to our parameters? Does this not just change the distribution of the prior? What if we were to construct a prior distribution that had the same shape as the hierarchical prior? 
- How to train hierarchical models? What is the narrow funnel problem? 

