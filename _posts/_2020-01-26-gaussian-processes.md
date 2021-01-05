---
layout: post
title: "Gaussian Processes and Regression"
date: 2020-12-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: A explanation of Gaussian processes, starting with simple intuition and building up to posterior inference. I sample from a GP in native Python and test GPyTorch on a simple simulated example.
---


I've found many articles about Gaussian processes that start their explanation by describing stochastic processes, then go on to say that a GP is a distribution over functions, or an infinite dimensional distribution. I find it harsh for an introduction. In this post I briefly explain GPs in a more approachable manner, and use code to show simulations from Gaussian processes. 

&nbsp;

### How to start thinking about a Gaussian Process?

We can start thinking about Gaussian processes by building up to it from univariate and multivariate Gaussians. Starting with a single random variable, $X_1 \sim N(\mu_1, \sigma_1^2)$, we can append another random variable $X_2 \sim N(\mu_2, \sigma_2^2)$ to get the vector $(X_1, X_2)$. If $X_1$ and $X_2$ have covariance $\sigma_{12}$, this vector will have distribution: 

$$ \begin{pmatrix} X_1 \\ X_2 \end{pmatrix} \sim N \left(\begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}, \begin{bmatrix} \sigma_1^2 & \sigma_{12} \\ \sigma_{12} & \sigma_2^2 \end{bmatrix} \right)$$

If we continue appending more Normally distributed random variables, $X_3, X_4, ...$ we can construct larger and larger multivariate Gaussians, once we have their mean and covariance. Then, the multivariate Gaussian will be fully specified by the mean and covariance matrix. To generalize this concept of continuously incorporating Normally distributed RV's into the same distribution, we need a function to describe the mean and another to describe the covariance.

This is what the Gaussian process provides. It is specified by a mean function, $\mu(X)$ and a covariance function (called the kernel function), $k(X, X')$, that returns the covariance between two points, $X$ and $X'$. Now we can model any amount (possibly infinite) of variables with the GP using the mean and kernel function. Since the GP can model an infinite number of random variables it is considered a distribution over functions, and written as: 

$$ f(x) \sim GP(\mu(x), k(x, x'))$$ 

The kernel function is simply a measure of how similar two random variables are, and an exmaple of one is the squared exponential kernel shown below. There are many more kernels and great writeups on them are [here](https://www.cs.toronto.edu/~duvenaud/cookbook/), [here](http://mlg.eng.cam.ac.uk/tutorials/06/es.pdf) and [here](https://statisticaloddsandends.wordpress.com/2019/06/28/common-covariance-classes-for-gaussian-processes/).

$$ k(x, x') = \sigma^2 \exp(-\frac{(x - x')^2}{2l^2}) $$


This is a mathematically loose intro to GP's, to convey the interpretation of "infinite dimensional" and "distribution over functions". The book [Gaussian Processes for Machine Learning](http://gaussianprocess.org/gpml/chapters/RW.pdf) goes into detail on the mathematics. 

&nbsp;

### Samples from a Gaussian Process Prior

Since the Gaussian process is essentially a generalization of the multivariate Gaussian, simulating from a GP is as simple as simulating from a multivariate Gaussian. The steps are below:

- Start with a vector, $x_1, x_2, ..., x_n$ that we will build the GP from. This can be done in Python with `np.linspace`. 
- Choose a kernel, $k$, and use it to calculate the covariance matrix for each combination of $(x_i, x_j)$. We should end up with a matrix of dimension $(n, n)$. This is the covariance matrix for the multivariate Gaussian we are sampling from. We'll use a zero-vector for its mean
- The resulting sample paths from this multivariate Gaussian are realizations of the Gaussian process, $GP(0, k)$. We can plot these values and a 95% confidence interval by taking the mean $\pm$ 1.96. 

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

# parametize a multivariate Gaussian with zero mean and K as the covariance matrix
ys = multivariate_normal.rvs(mean = np.zeros(n), 
                             cov = K, 
                             size = n_func)
```

<!-- <p align="center">
  <img src="/assets/gp_prior_samples.png" width="70%" height="70%">
</p> -->

<div class='figure' align="center">
    <img src="/assets/gp_prior_samples.png" width="70%" height="70%">
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> 
        7 samples from a Gaussian process prior, along with a 95% confidence interval 
    </div>
</div>

&nbsp;

### Gaussian Process + Regression

Nothing so far is groundbreaking, or particularly useful. All we have done is explained a way of generalizing the multivariate Normal, but haven't talked about how it can be used in real life. In order to do this, more background is needed to combine observed data with the GP.

To set the stage, imagine we are interested in modelling a function, $f$, for which we have noisy observations, $(x_i, y_i)$ where $x_i \in \R^D$. In typical Bayesian linear regression, we assume that $y$ is a linear function of $X$ given weights, $y = Xw$. Then we assign priors, $p(w)$, and build a posterior distribution for the weights, $p(w \mid y, X)$. This posterior is used to make future predictions and recreate $f = y + \epsilon$. 

$$ p(w \mid y_N, X_N) = \frac{p(y_N \mid X_N, w) p(w)}{p(y_N \mid X_N)} $$

However, in some cases we're only interested in making predictions, and in a Bayesian setting this boils down to 2 distributions: (1) the posterior predictive distribution in order to actually make a prediction and (2) the marginal likelihood for model comparison. 

$$ \text{Posterior predictive: } p(y_{n+1} \mid y_N, X_N) $$

$$ \text{Marginal likelihood: } p(y_{N} \mid X_N) $$

Expanding the formulations from Bayesian linear regression: 

$$ y = Xw \qquad \qquad \text{where: } w \sim N(0, \sigma_w^2) $$

And since $y$ is a linear function of $w$ (which is a random variable here), its prior is:

$$ y \sim N(0, \sigma_w^2 XX^T) $$

Accounting for noise in our observations, $\sigma^2_{err}$ the prior on our function, $f$, is: 

$$ f \sim N(0, \sigma_w^2 XX_T + \sigma^2_{err} I) $$



&nbsp;

#### Practical Problem

Building on GP's as a prior over functions, we can form a posterior distribution, $p(f \mid X, y)$ by conditioning on data. Intiutively, doing this excludes all functions that don't "pass through" our data, $(X, y)$. In this section we will use a Gaussian process prior to approximate a function. We'll also assume that there is no noise in our function observations, but this is obviously a terrible assumption in modelling real world systems.

I'll use [GPyTorch](https://gpytorch.ai/) for inference. There are easier ways to use GP's in Python but GPyTorch looks promising, especially with Pytorch integration.

Here's the function we want to approximate. The points in red are the training data, and we will try to approximate the blue section using a GP. 

```python
g = np.vectorize(lambda y: math.exp(-0.4 * y)*math.sin(4*y) + math.log(abs(y) + 1) + 1)
train_x = np.linspace(0, 4, 750)
test_x = np.linspace(4.01, 6, 100)
train_x = torch.tensor(train_x)
test_x = torch.tensor(test_x)

train_y = g(train_x) 
test_y = g(test_x) 
train_y=torch.tensor(train_y)
test_y=torch.tensor(test_y)

plt.figure(figsize=(6, 4), dpi=100)
sns.lineplot(train_x, train_y, color = 'red', label = "Train set")
sns.lineplot(test_x, test_y, color = 'blue', label = "Test set")
plt.title("Observed and test data")
plt.legend()
plt.show();
```

<p align="center">
  <img src="/assets/exactGP_simulated_function.png" width="70%" height="70%">
</p>

&nbsp;

#### Squared Exponential Kernel

Before we start, a big decision is the choice of kernel. This is prior specification in regular Bayesian modelling, but for a Gaussian process. The kernel we will be using here is the squared exponential kernel / radial basis function kernel / Gaussian kernel. 

$$ k(x, x') = \sigma_f^2 \exp(\frac{(x - x')^2}{2 \iota^2}) $$

There are two parameters in this kernel that need to be estimated from the data, and David Duvenaud describes them well [here](https://www.cs.toronto.edu/~duvenaud/cookbook/): 

- The lengthscale $\iota$ determines the length of the 'wiggles' in your function. In general, you won't be able to extrapolate more than $\iota$ units away from your data.
- The output variance $\sigma_f^2$ determines the average distance of your function away from its mean.

&nbsp;

```python
class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() # mean
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) # kernel

    def forward(self, x):
        mean_x = self.mean_module(x) 
        covar_x = self.covar_module(x) 
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGP(train_x, train_y, likelihood)
```

&nbsp;

<p align="center">
  <img src="/assets/squared_exp_kernel_posterior.png" width="100%" height="70%">
</p>


<!-- 
#### Other Kernels

There are a ton of other kernels, and it'll be interesting to see what their posterior samples look like. 

##### Marten Kernel 


<p align="center">
  <img src="/assets/marten_kernel_posterior.png" width="100%" height="70%">
</p>


<p align="center">
  <img src="/assets/periodic_kernel_posterior.png" width="100%" height="70%">
</p> -->




<!-- ### Questions

- How are the weights, $w$ integrated out when doing inference on a GP?
- Can I use GPyTorch for a text classification model with TF-IDF features?
- What does it mean to "fit a Gaussian process"? What is actually going on in the background? I don't understand how we can simulate draws from the prior.
- Imagine points on a line. If we divide the line into 5 equal points and each point is Normally distributed, this is what a multivariate gaussian would look like, however if we wanted every single one of the points on the line to be normally distributed, this is what a guassian process would look like.
- Can I make an active learner using a GP and the embeddings from a NN to learn  -->


<!-- ## A Note on Regression

Let's start by explaining different types of linear regression. In simple linear regression, we first make a linearity assumption about the data (we assume the target variable is a linear combination of the features), then we estimate model parameters based on the data. In Bayesian linear regression, we make the same linearity assumption, however we take it a step further and make an incorporate beliefs about the parameters into the model (priors), then learn the parameters from the data.
Gaussian Process Regression takes a different approach. We don't drop the linearity assumption, and the priors on the parameters. Instead we put a prior on **_all possible models_**. As we observe data, the posterior.

**What is Gaussian Process Regression?** - In Gaussian Process regression, a GP is used as a prior on $f$. This means that the posterior distribution over functions is also a GP. The posterior has to be updated every time we observe new data, because the specification of the posterior depends on observed data. Intuitively, the reason we update the GP is to eleminate all functions that do not pass through the observed data points.

### Notes

- The GP is a prior over functions. It is a prior because we specify that we want smooth functions, and we want our points to be related in a certain way, which we do with the kernel. -->