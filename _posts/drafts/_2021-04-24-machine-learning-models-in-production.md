---
layout: post
title: "Machine Learning Models in Production"
date: 2021-01-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Private post for me to remember the models I built that are currently in production. Writing this so that after grad school, I will still remember this workflow and the painpoints. 
---

# Document Type Classifier - Sorting Hat

In chronological order: 

- Defining classes
- Gathering data using external sources and active learning 
- Building initial model
- Determining that there should be another class
- Gathering data for this new class
- Fumbling around trying to organzie labeled data for pages and documents 
- More fumbling trying to evaluate both the page model and the baseline classification model
- Taking a step back to clean the data and collect some more 
- Oh! This is working a lot better than iterating on model architectures
- How do we test this on live data? I guess we have to do this annotation ourselves. 
- Let's try Figure8... ok this simple task is too difficult for them 
- Time to serve model, use MLflow to package it
- Run on some docs in production and evaluate results
- Tweak model based on results 


# Short Document Classifier 

In chronological order:

- We have data, but not for short documents 
- Split up documents into chunks based on word counts 
- Start building a baseline model. Can't use SVM as final model because we need confidence scores
- Every model is producing very low confidence scores, can we change this?
- Start building Bayesian hierarchical model 
- Can collect 500 docs every 2 weeks, need to determine what classes the model is performing poorly in 