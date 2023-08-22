---
layout: post
title: "(not so) Simple Linear Regression"
date: 2023-07-09 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Poking the linear regression model
---


Questions to answer: 

- What are the assumptions of linear regression
- What happens when each of these assumptions are broke. With proofs, code and explanations
- Can the $R^2$ get larger with penalized regression? Eg: ridge/lasso 
- What happens to regression estimates when the features have high multicollinearity? Proof
- What happens to the $R^2$ when we add another covariate to our regression?
- What does the condition number tell us and how can we use it?
- Multicollinearity affects the inversion of the feature matrix. If the determinant is very close to 0, the determinant of the inverted matrix will become extremely large. This will multiply the noise in the data. How does this work? 
- A stock's $\beta$ is calculated by regressing its returns onto the S&P 500 returns. What is the change in interepretation when we flip this regression and why is it like this? 
- Eigenvalues are not uniquely determined. How does this affect linear regression and PCA? 
- What are the differences between linear regression and the weights of the first principal component?
- 