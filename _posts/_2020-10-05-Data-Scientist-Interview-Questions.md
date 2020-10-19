---
layout: post
title: "Collection of Cool Data Scientist Interview Questions"
date: 2020-09-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: A collection of cool data scientist interview questions that I've come across
---


### Machine Learning
- What is the difference between K-mean and EM?
- Why use feature selection? If two predictors are highly correlated, what is the effect on the coefficients in the logistic regression? What are the confidence intervals of the coefficients?
- What is the function of p-values in high dimensional linear regression?
- Describe linear regression to a child, to a first-year college student, and to a seasoned mathematician.
- What is the difference between MLE and MAP?
- What are the assumptions of linear regression?
- What algorithm would you use to predict if a driver will accept a ride request or not? What features would you use?
- What's the difference between Supervised vs. Unsupervised machine learning?
- What is precision? What is specificity? What is sensitivity/recall?
- How do you evaluate the performance of a regression prediction model as opposed to a classification prediction model?
- How do you deal with missing values?
- What are the relationships between the coefficient in the logistic regression and the odds ratio?
- How do you validate a machine learning model?
- What's your favorite kernel function?
- What’s the difference between l1 and l2 regularization and why would you use each?
- What features would you use to predict the time spent for a restaurant preparing food from the moment an order comes in?
- Can you come up with a scenario in which you would rather under-predict versus over-predict?
- Analyzing the results of a model, how would you explain the tradeoff between bias and variance?
- Explain how a Random Forest model actual works under the hood.
- How do you know if you have enough data for your model?
- What are the differences between L1 and L2 regularization, why don’t people use L0.5 regularization for instance?
- Write the equation for building a classifier using Logistic Regression.
- Why is Rectified Linear Unit a good activation function?
-


### Probability and Statistics
- For a sample size of N, the margin of error is 3. How many more samples do we need for the margin of error to hit 0.3?
- What is the assumption of error in linear regression?
- How can you tell if a given coin is biased?
- Explain how a probability distribution could be not normal and give an example scenario.
- You have a deck and you take one card at random and guess what the card is. What is the probability you guess right?
- What is the difference between parametric and non-parametric testing?
- Given a random Bernoulli trial generator, write a function to return a value sampled from a normal distribution.
- Given three random variables independent and identically distributed from a uniform distribution of 0 to 4, what is the probability that the median is greater than 3?
- What is a standard deviation?
- What is the difference between precision/specificity?
- Define a confidence interval?
- How do you generate a uniform number using a nonuniform distributed function?
- Write a function to sample from a multinomial distribution.
- Given uniform distributions X and Y and the mean 0 and standard deviation 1 for both, what’s the probability of 2X > Y?
- There are four people in an elevator and four floors in a building. What’s the probability that each person gets off on a different floor?
  - What’s the probability that two people get off on the same floor?
- Given a deck of cards labeled from 1 to 100, what’s the probability of getting Pick 1 < Pick2 < Pick3?



### Hypothesis Testing and Product
- How would you detect inappropriate content on Youtube?
- How do you test if a new feature has increased engagement in Google's ecosystem?
- If the outcome of an experiment results in one group clicking 5% more than the other, is that a good result?
- Let’s say we launch a new Uber Eats feature. What would you choose as the key metric?
- How would you design an incentive scheme for drivers such that they would more likely go into city areas where demand is high?
- What metrics would you use to track Uber’s strategy of using paid advertising to acquire customers’ works? How would you figure out an acceptable cost of customer acquisition?
- What are the costs of having a fleet of vehicles take Google street view photos of every major city in the US every day?
- Explain the importance of a p-value?
- What is a p-value?
- How would you grow LinkedIn messaging?
- LinkedIn wants to release a new auto-complete messaging feature in Inmail. How would you measure the success of the feature?
- Let’s say we’re given a dataset of page views where each row represents one page view. How would you differentiate between scrapers
- Due to engineering constraints, the company can’t AB test a feature before launching it. How would you analyze how the feature is performing?
- Let’s say at LinkedIn we want to implement a green dot for an “active user” on the new messaging platform. How would you analyze the effectiveness of it for roll out?
- How do you know if you have enough sample size?
- How do you run significance tests on more than one variant?
- How do you reduce variance and bias in an AB test?
- Given a month’s worth of login data from Netflix such as account_id, device_id, and metadata concerning payments, how would you detect payment fraud?
- How would you design an experiment for a new content recommendation model we’re thinking of rolling out? What metrics would matter?
- How would you select a representative sample of search queries from five million?
- If Netflix is looking to expand its presence in Asia, what are some factors that you can use to evaluate the size of the Asia market, and what can Netflix do to capture this market?
- 



### Programming
- Build a text wrapper. For example, split a long sentence by some character limit only at the spaces.
- Write a production code to find all combinations of numbers in a list that sum up to 8.
- What do nested SELECT and WITH do in SQL?
- Estimate the value of Pi using the Monte Carlo Algorithm.
- How do you implement Fibonacci in python? Why is loop is better than recursion?
- Give an array of unsorted random numbers (decimals), find the interquartile distance.
- Given a list of all followers in format: 123, 345;234, 678;345, 123;…where the first column contains the ID of the follower, and the second one is the ID of who’s followed, find all mutual follows(pair 123, 345 in the example above). Do the same in the case, when this list does not fit into the memory.
- Given a list of characters, a list of prior of probabilities for each character, and a matrix of probabilities for each character combination, return the optimal sequence for the highest probability.
- Write a function that can take a string and return a list of bigrams.
-

### SQL
- Given a payment transactions table and a customers table, return the customer’s name and the first transaction that the customer made.
- Given a payments transactions table, return a frequency distribution of the number of payments each customer made. (I.E. 1 transaction — 100 customers, 2 transactions — 50 customers, etc…)
- Given the same payments table, return the cumulative distribution. (At least one transaction, at least two transactions, etc…)
- Given a table of — friend1|friend2. Return the number of mutual friends between two friends.
-
