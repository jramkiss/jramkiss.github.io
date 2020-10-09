---
layout: post
title: "Machine Learning Interview Prep"
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

Decision trees split the input space into non-linear regions by minimizing cross-entropy loss between regions. Let $p_c$ be the proportion of samples in region $R$ that are in class $c$, then:

$$ L_{cross}(R) = - \sum_c \hat{p}_c \text{log}(\hat{p}_c) $$

Since cross-entropy is strictly concave, as long as $p_1 \ne p_2$ and regions $R_1$ and $R_2$ are non-empty, then the weighted sum of children losses will always be less than the parent (more on this [here](http://cs229.stanford.edu/notes/cs229-notes-dt.pdf)). This can be a problem for overfitting, which is why we regularize the tree:

- Minimum leaf size: Do not split $R$ if its cardinality (number of points) is below a threshold
- Max depth: Do not split $R$ if more than a fixed number of splits were already taken
- Max nodes: Stop after a fixed number of nodes

A main problem with decision trees apart from overfitting is that it struggles to capture additive structure in data. This is demonstrated in the image taken from [here](http://cs229.stanford.edu/notes/cs229-notes-dt.pdf), below:

<p align="center">
  <img src="/assets/decision-tree-additive-structure.png" width="100%" height="100%">
</p>

Fully grown decision trees are high-variance and low-bias, so bagging can help. A downside to bagging is that we lose the interpretability we had with the single decision tree. However this can be somewhat accounted for by measuring variable importance by counting the number of times variables are split on.
Bagged decision trees can be taken one step further by only considering a subset of features at each split. This furhter reduces variance but increases bias and is called **random forests**.

#### Gradient Boosted Machines

#### Support Vector Machines

SVM is a linear classifier that finds the optimal margin of seperation between two classes. It maximizes the orthogonal distance between each point and a linear boundary, eventually reducing the dataset into a couple important points close to the margin called support vectors. These are the only datapoints that influence future predictions.

The linear boundary is parameterized by $w, b$ and the target variable, $y$ is one of $(-1, 1)$:
$$ h(x) = g(w^Tx+b)$$

Where $h(x) = 1$ if $w^Tx+b \ge 0$ and $h(x) = -1$ otherwise. We define the functional margin of $(w, b)$ to be the orthogonal distance between the boundary and $x_i$. All this means is the shortest distance between the bounary and $x_i$:
$$\gamma_i = y_i (w^Tx_i + b) $$

Unlike logistic regression, SVM has no notion of confidence. All we get in return from the model is a predicted class.

We are looking for the optimal boundary separating the positive and negative training examples with a gap of at least $\gamma$. In addition, we need to constrain $||w|| = 1$ so that we can't arbitrarily scale $(w, b)$. This constraint makes the optimization non-convex and difficult, so instead we change the constraint to: $\hat{\gamma} = 1$, which results in:

$$
\underset{w, b}{\text{max}}  \frac{1}{2}||w||^2 \\
\text{s.t.} \\ y_i(w^Tx_i + b) \ge 1 \\
$$

Now suppose we want to make a prediction at a new point, $x$. We would calculate $w^Tx + b$ and predict $y = 1$ if and only if this quantity is bigger than zero. From the Lagrangian solution, we have that:

$$\begin{aligned}
w^Tx + b &= (\sum_{i=1}^{m} \alpha_i y_i x_i)^Tx + b \\
&= (\sum_{i=1}^{m} \alpha_i \langle y_i, x_i\rangle)^Tx + b
\end{aligned}
$$

And the parameters $\alpha_i$ are zero everywhere except for the support vectors (points on the boundary). This means that only the support vectors affect the classification of new points, and the model is efficient in high dimensional spaces with lots of data, as we only need to find the inner products between $x$ and the support vectors.

This inner product opens the door for Kernels. Kernels induce additional feature maps into the data, which can have the notion of increased dimenionality, making non-linearly seperable data linearly seperable.

So far we have only mentioned hard margin SVM, where we don't allow data inside the separating margin. We can relax this constraint by penalizing points that lie inside of the margin using the $L_1$ norm. The new optimization problem becomes:

$$
\underset{w, b}{\text{max}}  \frac{1}{2}||w||^2 + X\sum_{i=1}^{m}\epsilon_i\\
\text{s.t.} \\ y_i(w^Tx_i + b) \ge 1 - \epsilon_i \\
\epsilon_i \ge 0
$$

More on SVM's [here](http://cs229.stanford.edu/notes/cs229-notes3.pdf).

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
http://cs229.stanford.edu/notes/cs229-notes-all/error-analysis.pdf

We can forget about bias and variance, and just focus on building a model given some training data. We know that there is a possibility that we overtrain the model on this specific set of training data, so we're careful not to do so. On the flip side, we're aware that if we train too little, the model will predict garbage as it isn't able to capture the signal. This tradeoff between overfitting and underfitting is exactly the bias-variance tradeoff.
If we over train the model too our training set, when we try to generalize to more data we can expect varying predictions, i.e. high variance. However if we undertrain the model, we can expect more consistent, but wrong predictions.


#### Regularization
http://cs229.stanford.edu/notes/cs229-notes5.pdf

#### Bagging, Boosting, Gradient Boosting

###### Bagging
Bagging stands for Bootstrap Aggregation and is a variance reduction technique. We start by bootstrap sampling a dataset into $M$ bootstrap samples. Then we build models, $g_m(X)$, on each sample and aggregate their outputs to make predictions: $g(X) = \sum_m\frac{g_m(X)}{M}$.

Bagging creates less correlated predictors than if we just trained on the whole sample, thus reducing overall variance. However the overall bias is increased because each individual bootstrap sample doesn't contain the full training set. More on bagging [here](http://cs229.stanford.edu/notes/cs229-notes-ensemble.pdf).

###### Boosting
Boosting is a bias-reduction technique where we iteratively train weak learners (ex: decision trees with only 1 split / decision stumps). At each iteration we give misclassified points higher weight and train another learner. At the end we have an ensemble of weak learners.
More on boosting [here](http://cs229.stanford.edu/extra-notes/boosting.pdf).

###### Gradient Boosting

Don't fully understand, read about it: [http://cs229.stanford.edu/notes/cs229-notes-ensemble.pdf](http://cs229.stanford.edu/notes/cs229-notes-ensemble.pdf)

#### Precision, Recall and F1 Score

When evaluating a classifier, we are interested in two things:

1) When the model predicts Class A (precision), what % of those predictions are correct?
2) What % of data in Class A did the model correctly predict (recall) in Class A?

There are cases where one metric is more important than the other. In a spam model for example, we would favour Recall over Precision as we don't want to let spam through. However in models for medical diagnosis, Precision can be favoured over Recall, as we want to limit the number of false positive diagnoses.

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
