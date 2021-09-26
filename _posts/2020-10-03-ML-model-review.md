---
layout: post
title: "ML Models Reference Sheet"
date: 2020-09-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Originally written to be a personal refresher for some ML models and concepts.
---


## Resources

- https://maria-antoniak.github.io/2018/11/19/data-science-crash-course.html

## Models

#### Linear Regression, Logistic Regression and GLM's
http://cs229.stanford.edu/notes/cs229-notes1.pdf

---


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

---

#### Support Vector Machines

SVM is a linear classifier that finds the optimal margin of seperation between two classes. It maximizes the orthogonal distance between each point and a linear boundary, eventually reducing the dataset into a couple important points close to the margin called support vectors. These are the only datapoints that influence future predictions.

The linear boundary is parameterized by $w, b$ and the target variable, $y$ is one of $(-1, 1)$:
$$ h(x) = g(w^Tx+b)$$

Where $h(x) = 1$ if $w^Tx+b \ge 0$ and $h(x) = -1$ otherwise. We define the functional margin of $(w, b)$ to be the orthogonal distance between the boundary and $x_i$. All this means is the shortest distance between the bounary and $x_i$:
$$\gamma_i = y_i (w^Tx_i + b) $$

Unlike logistic regression, SVM has no notion of confidence. All we get in return from the model is a predicted class.

We are looking for the optimal boundary separating the positive and negative training examples with a gap of at least $\gamma$. In addition, we need to constrain $\text{||}w\\text{||} = 1$ so that we can't arbitrarily scale $(w, b)$. This constraint makes the optimization non-convex and difficult, so instead we change the constraint to: $\hat{\gamma} = 1$, which results in:

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

---


#### k-Means and k-NN
k-Means: http://cs229.stanford.edu/notes/cs229-notes7a.pdf

#### Principal Component Analysis

Consider a dataset $\{x_i; i = 1..m\}$ that represent pilots of RC helicopters. We can have that $x_{i1}$ measures the pilot's skill and $x_{i2}$ measures thow much he/she enjoys flying. Since RC helicopters are usually difficult to fly, the most skilled pilots are the ones who enjoy flying the most. Therefore we can expect that the data actually lies on a diagonal axis representing the "piloting karma" of a person. And orthogonal to that axis is some noise.
In addition to this, we may want an automatic way of detecting when 2 covariates have high covariance, so that we can either combine them or exclude them.

<p align="center">
  <img src="/assets/PCA-pilot-example.png" width="80%" height="70%">
</p>

We want to find the directions of the underlying data, $u_1$ and $u_2$. Given a unit vector $u$ and a point, $x$, the length of the projection of $x$ onto direction $u$ is $x^Tu$. i.e. if $x^{(i)}$ is a point in our dataset and we project it onto direction $u$, it is $x^Tu$ distance from the origin. Hence, to maximize the variance of projections, we want to choose $u$ to maximize:

$$\begin{aligned}
\frac{1}{m} \sum_{i=1}^{m} (x_{i}^Tu)^2 &= \frac{1}{m} \sum_{i=1}^{m} u^T x_i x_i^T u \\
&= u^T (\frac{1}{m} \sum_{i=1}^{m}x_i x_i^T) u
\end{aligned}$$

Maximizing this subject to $||u||_2=1$ gives the principal eigenvector of $\Sigma = \frac{1}{m} \sum_{i=1}^{m} x_i x_i^T$, which is the empirical covariance of the data, assuming the mean is $0$. More generally, if we wish to project our data into a $k$-dimensional subspace $(k < n)$, we should choose $u_1, ..., u_k$ to be the top $k$ eigenvectors of $\Sigma$.

More on PCA [here](http://cs229.stanford.edu/notes/cs229-notes10.pdf).


#### Convolutional Neural Networks
http://d2l.ai/chapter_convolutional-neural-networks/index.html


## Concepts

#### Precision and Recall

- Precision is the fraction of points predicted to be class $k$ that are correct
- Recall is the fraction of class $k$ that are predicted correctly 

High precision low recall example: We have 10 data points in class $k$ and run our classifier on them. It predicts 1 point to be in class $k$ and the others in class $j$. Then the recall is 10% but the precision is 100%.

When evaluating a classifier, we are interested in two things:

1) When the model predicts Class A (precision), what % of those predictions are correct?
2) What % of data in Class A did the model correctly predict (recall) in Class A?

There are cases where one metric is more important than the other. In a spam model for example, we would favour Recall over Precision as we don't want to let spam through. However in models for medical diagnosis, Precision can be favoured over Recall, as we want to limit the number of false positive diagnoses.


#### k-Fold Cross Validation

Cross validation is simply holding out part of out data to be used for testing. In its most basic form, 1-fold cross validation, we make 1 split in the data and use part for training and the rest for testing, however we can extend this to multiple splits. 
We can use k-fold cross validation to determine the robustness of our model. Here we divide the data into $k$ "folds", train on $k-1$ and evaluate on the $k$th. Be careful to do any data cleaning / transformations on the $k-1$ folds so that you don't allow data leakage.

#### Central Limit Theorem

The CLT states that with a population with mean, $\mu$ and standard deviation, $\sigma$. If we take sufficiently large samples with replacement then the distribution of the sample means will be approximately Normally distributed.

In practice it helps us assign a distribution for sample estimates, which are used in hypothesis testing.


#### Bias / Variance Tradeoff
http://cs229.stanford.edu/notes/cs229-notes4.pdf
http://cs229.stanford.edu/notes/cs229-notes-all/error-analysis.pdf

We can forget about bias and variance, and just focus on building a model given some training data. We know that there is a possibility that we overtrain the model on this specific set of training data, so we're careful not to do so. On the flip side, we're aware that if we train too little, the model will predict garbage as it isn't able to capture the signal. This tradeoff between overfitting and underfitting is exactly the bias-variance tradeoff.
If we over train the model too our training set, when we try to generalize to more data we can expect varying predictions, i.e. high variance. However if we undertrain the model, we can expect more consistent, but wrong predictions.

Bias is the model's tendency to consistently learn the same wrong thing, whereas variance is the tenddedncy to consistently learn random things. Simple models have high bias because they are not able to properly induce the properties of the data, however more complex models have a tendency to have high variance because their hypothesis spaces are much larger. This can be thought of as simple models only considering a small set of possible outcomes (ex of an outcome is a linear boundary), however complex models consider a much larger set of possible outcomes (neural nets can represent any function).


#### Coefficient of Determination, $R^2$ and Adjusted $R^2$

$R^2$ is the percentage of variance in the target variable explained by the covariates. The variance in the target variable can be thought of as variance along the $y$ axis. Low variance in the target variable would imply data points are close to the regression line, and vice versa for high variance. High variance likely means the covariates are inappropriate for the problem and that there are other drivers of the target variable. 

$$R^2 = \frac{\text{Variance explained by the model}}{\text{Total variance}}$$

$R^2$ necessarily increases as we add more covariates. Ideal behaviour would be for us to penalize additional covariates, which is what adjusted $R^2$ does. 



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
