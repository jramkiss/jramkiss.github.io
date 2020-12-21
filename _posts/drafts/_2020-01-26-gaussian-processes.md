---
layout: post
title: "Gaussian Processes"
date: 2020-01-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Explanation of Gaussian processes for dummies like myself. Starting with how to think about a Gaussian process, then moves on to simulation from Gaussian process priors and ends with Gaussian process regression 
---


### Why is this a cool model?

Imagine learning about neural networks for the first time, and thinking _"wow, is there anything this thing can't do??"_. And the heartbreak later on, when you realize a [neural network trained to classify cats and dogs](https://jramkiss.github.io/2020/07/29/overconfident-nn/) predicts you're a dog with 98% confidence. Clearly they aren't _that_ good, since they don't even know what they don't know. This is why I find Gaussian Processes so cool, they are the gold standard for "knowing when you don't know".

&nbsp;

I've found many articles about Gaussian processes that start their explanation by describing stochastic processes, then go on to say that a GP is a distribution over functions, or an infinite dimensional distribution. This may be the "right" way to approach it, but I find it harsh for an introduction. In this post I try to explain GPs in a more approachable manner, and use code to show how simple simulation from GPs are.

&nbsp;

### How to start thinking about Gaussian Processes?

One useful way to start thinking about Gaussian processes is to build up from the relationship between the univariate and multivariate Gaussian distributions. Starting with a single random variable, $X_1 \sim N(\mu_1, \sigma_1^2)$, we can append another random variable $X_2 \sim N(\mu_2, \sigma_2^2)$ to obtain a multivariate Gaussian for $(X_1, X_2)$. If $X_1$ and $X_2$ have covariance $\sigma_{12}$, this distribution will look like: 

$$ \begin{pmatrix} X_1 \\ X_2 \end{pmatrix} \sim N \left(\begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}, \begin{bmatrix} \sigma_1^2 & \sigma_{12} \\ \sigma_{12} & \sigma_2^2 \end{bmatrix} \right)$$

If we continue appending more Normally distributed random variables, $X_3, X_4, ...$ we can continue to construct larger and larger multivariate Gaussians, all we need is their mean and covariance with all other variables. To generalize this concept of continuously incorporating Normally distributed RV's into the same distribution, we need a function to describe the mean and another to describe the covariance.

This is exactly what the Gaussian process provides. It is specified by a mean function, $\mu(X)$ and a covariance function (called the kernel function), $k(X, X')$, that returns the covariance between any two points, $X$ and $X'$. Now we can model any amount (possibly infinite) of variables with the GP using the mean and kernel function. Since the GP can model an infinite number of random variables, this is why GP's are considered a distribution over functions, and it is written as: 

$$ f(x) \sim GP(\mu(x), k(x, x'))$$ 

An exmaple of the mean and kernel functions are: 

$$ \mu(x) = 0  $$

$$ k(x, x') = \sigma^2 \exp(-\frac{(x - x')^2}{2l^2}) $$

This is the squared exponential kernel. There are many more kernels and a great writeup on them can be found [here](https://www.cs.toronto.edu/~duvenaud/cookbook/), [here](http://mlg.eng.cam.ac.uk/tutorials/06/es.pdf) and [here](https://statisticaloddsandends.wordpress.com/2019/06/28/common-covariance-classes-for-gaussian-processes/).


&nbsp;

### Samples from a Gaussian Process Prior

Since the Gaussian process is, for practical purposes, a generalization of the multivariate Gaussian, simulating from a GP is as simple as simulating from a multivariate Gaussian. The steps are below:

- Start with a vector, $x_1, x_2, ..., x_n$. This can be done in Python with `np.linspace`
- Use the kernel to calculate the covariance matrix for each combination of $(x_i, x_j)$. We should end up with a matrix of dimension $(n, n)$. This matrix will act as the covariance matrix for the multivariate Gaussian we are sampling from. We'll also use a zero-vector for the mean
- The resulting sampled paths from this multivariate Gaussian are realizations of the Gaussian process. We can plot these values and a 95% confidence interval by taking the mean $\pm$ 1.96. 

Code to do this is below: 

```python
kernel = 1.0 * RBF(1.0)

n = 100 
n_func = 7 # number of functions to sample from the GP 
L = -5; U = 5

# start with X = (x_1, x_2, ..., x_n)
X = np.linspace(L, U, n).reshape(-1, 1)

#  use kernel to calculate the covariance matrix
K = kernel(X)

# use the covariance matrix and zero-vector for the mean to parametize a multivariate Gaussian
ys = multivariate_normal.rvs(mean = np.zeros(n), 
                             cov = K, 
                             size = n_func)
```


### Gaussian Process Regression

So far we have seen how to think of GP's as a prior over functions, and how to sample functions from a GP. The next step is to use a GP as a prior for a modelling problem. In this case, we use 

### Questions

- Test out Gaussian Processes on high dimensional data where we don't have the ability to plot. 
- What does it mean to "fit a Gaussian process"? What is actually going on in the background? I don't understand how we can simulate draws from the prior.
- Imagine points on a line. If we divide the line into 5 equal points and each point is Normally distributed, this is what a multivariate gaussian would look like, however if we wanted every single one of the points on the line to be normally distributed, this is what a guassian process would look like.
- Can I make an active learner using a GP and the embeddings from a NN to learn 


<!-- ## A Note on Regression

Let's start by explaining different types of linear regression. In simple linear regression, we first make a linearity assumption about the data (we assume the target variable is a linear combination of the features), then we estimate model parameters based on the data. In Bayesian linear regression, we make the same linearity assumption, however we take it a step further and make an incorporate beliefs about the parameters into the model (priors), then learn the parameters from the data.
Gaussian Process Regression takes a different approach. We don't drop the linearity assumption, and the priors on the parameters. Instead we put a prior on **_all possible models_**. As we observe data, the posterior.

**What is Gaussian Process Regression?** - In Gaussian Process regression, a GP is used as a prior on $f$. This means that the posterior distribution over functions is also a GP. The posterior has to be updated every time we observe new data, because the specification of the posterior depends on observed data. Intuitively, the reason we update the GP is to eleminate all functions that do not pass through the observed data points.

### Notes

- The GP is a prior over functions. It is a prior because we specify that we want smooth functions, and we want our points to be related in a certain way, which we do with the kernel. -->