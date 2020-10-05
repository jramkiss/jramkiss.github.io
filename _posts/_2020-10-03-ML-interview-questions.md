---
layout: post
title: "Machine Learning Interview Questions and Answers"
date: 2020-09-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Designed to be a personal reference guide for ML intreview questions
---

I break down the ML interview into 3 sections:

- Models
- Concepts
- Situations


## Models

#### Linear Regression, Logistic Regression and GLM's
http://cs229.stanford.edu/notes/cs229-notes1.pdf

#### Decision Trees and Random Forests
http://cs229.stanford.edu/notes/cs229-notes-dt.pdf

#### Gradient Boosted Machines

#### Support Vector Machines

SVM is a linear classifier that tries to find the optimal margin of seperation between two classes. The linear classifier is parameterized by $w, b$ and the target variable, $y$ is one of $(-1, 1)$:
$$ h(x) = g(w^Tx+b)$$

Where $h(x) = 1$ if $w^Tx+b \ge 0$ and $h(x) = -1$ otherwise. We define the functional margin of $(w, b)$ to be:
$$\gamma_i = y_i (w^Tx_i + b) $$

So if $y_i = 1$, for the functional margin to be large $w^Tx_i + b$ needs to be large. Also, if $y_i(w^Tx_i + b) > 0$ then the exmaple is correct.

We are looking for the optimal classifier that separates the positive and negative training examples with a gap of at least $\gamma$. In addition, we need to constrain $||w|| = 1$ so that we can't arbitrarily scale $(w, b)$. This constraint makes the optimization non-convex and difficult, so instead we change the constraint to: $\hat{\gamma} = 1$, which results in:

$$
\underset{w, b}{\text{max}}  \frac{1}{2}||w||^2 \\
\text{s.t.} \\ y_i(w^Tx_i + b) \ge 1 \\
$$

Suppose we want to make a prediction at a new point, $x$. We would calculate $w^Tx + b$ and predict $y = 1$ if and only if this quantity is bigger than zero. From the Lagrangian solution, we have that:

$$\begin{aligned}
w^Tx + b &= (\sum_{i=1}^{m} \alpha_i y_i x_i)^Tx + b \\
&= (\sum_{i=1}^{m} \alpha_i \langle y_i, x_i\rangle)^Tx + b
\end{aligned}
$$

And the parameters $\alpha_i$ are zero everywhere except for the support vectors (points on the boundary). This means that only the support vectors affect the classification of new points, and the model is efficient in high dimensional spaces with lots of data, as we only need to find the inner products between x and the support vectors.

This inner product opens the door for Kernels. Kernels induce additional feature maps into the data, which can have the notion of increased dimenionality, making non-linearly seperable data linearly seperable.

So far we have only mentioned hard margin SVM, where we don't allow data inside the separating margin. We can relax this constraint by penalizing points that lie inside of the margin using the $L_1$ norm. The new optimization problem becomes:

$$
\underset{w, b}{\text{max}}  \frac{1}{2}||w||^2 + X\sum_{i=1}^{m}\epsilon_i\\
\text{s.t.} \\ y_i(w^Tx_i + b) \ge 1 - \epsilon_i \\
\epsilon_i \ge 0
$$


#### Naive Bayes

http://cs229.stanford.edu/notes/cs229-notes2.pdf

#### k-Means and k-NN
k-Means: http://cs229.stanford.edu/notes/cs229-notes7a.pdf

#### Principal Component Analysis
http://cs229.stanford.edu/notes/cs229-notes10.pdf

#### Convolutional Neural Networks

#### Sequence Models and LSTM's


## Concepts

#### Bias / Variance Tradeoff
http://cs229.stanford.edu/notes/cs229-notes4.pdf

#### Regularization
http://cs229.stanford.edu/notes/cs229-notes5.pdf

#### Boosting, Bagging, Gradient Boosting
http://cs229.stanford.edu/notes/cs229-notes-ensemble.pdf

#### Precision, Recall and F1 Score

#### Activation Functions

#### Dropout

#### Batch Normalization

#### Residual and Dense Connections

#### Attention


## Situations

#### First steps in model building

#### Dealing with imbalanced classes

####


## Advanced

#### EM Algorithm
http://cs229.stanford.edu/notes/cs229-notes8.pdf

#### Factor Analysis
http://cs229.stanford.edu/notes/cs229-notes9.pdf
