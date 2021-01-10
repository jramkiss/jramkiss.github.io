---
layout: post
title: "Gaussian Processes and Regression"
date: 2021-01-05 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
image: /assets/squared_exp_kernel_posterior.png
summary: A explanation of Gaussian processes and Gaussian process regression, starting with simple intuition and building up to inference. I sample from a GP in native Python and test GPyTorch on a simple simulated example.
---


I've found many articles about Gaussian processes that start their explanation by describing stochastic processes, then go on to say that a GP is a distribution over functions, or an infinite dimensional distribution, etc. etc. I find these harsh for an introduction, so in this post I try to explain GPs in a more approachable manner. Then I talk about Gaussian process regression and its relationship with Bayesian linear regression, with a simple example using GPyTorch.

&nbsp;

### How to start thinking about a Gaussian Process?

We can start by building multivariate Gaussians from univariate Gaussians. With a single random variable, $X_1 \sim N(\mu_1, \sigma_1^2)$, we can append $X_2 \sim N(\mu_2, \sigma_2^2)$ to get the vector $(X_1, X_2)$. If $X_1$ and $X_2$ have covariance $\sigma_{12}$, this vector will have distribution: 

$$ \begin{pmatrix} X_1 \\ X_2 \end{pmatrix} \sim N \left(\begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}, \begin{bmatrix} \sigma_1^2 & \sigma_{12} \\ \sigma_{12} & \sigma_2^2 \end{bmatrix} \right)$$

If we continue appending more Normally distributed random variables, $X_3, X_4, ...$ constructing larger multivariate Gaussians is easy, once we have their mean and covariance. Then, this multivariate Gaussian will be fully specified by the mean vector and covariance matrix. To generalize this concept of continuously incorporating Normally distributed RV's into the same distribution, we need a function to describe the mean and another to describe the covariance.

This is what the Gaussian process provides. It is specified by a mean function, $\mu(x)$ and a covariance function (called the kernel function), $k(x, x')$, that returns the covariance between two points, $x$ and $x'$. Now we are not limited to $n$ variables for a $n$-variate Gaussians, but can model any amount (possibly infinite) with the GP. We write: 

$$ f(x) \sim GP(\mu(x), k(x, x'))$$ 

The kernel function, $k(x, x')$ is simply a measure of how similar $x$ and $x'$ are, and an exmaple of one is the squared exponential kernel:

$$ k(x, x') = \sigma^2 \exp(-\frac{(x - x')^2}{2l^2}) $$

This is a loose intro to GP's to convey the interpretation of "infinite dimensional" and "distribution over functions". The book [Gaussian Processes for Machine Learning](http://gaussianprocess.org/gpml/chapters/RW.pdf) goes into detail. 

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

<div class='figure' align="center">
    <img src="/assets/gp_prior_samples.png" width="70%" height="70%">
    <div class='caption' width="70%" height="70%">
        <!-- <span class='caption-label'>Figure 1.</span>  -->
        <p> 7 samples from a Gaussian process prior, along with a 95% confidence interval. Each curve is the result of sampling from a multivariate Gaussian with $n=100$ variables. If we reduce $n$, the samples will look less and less smooth, until $n=2$, where the sample will just be a line. </p>
    </div>
</div>

&nbsp;

### Gaussian Process + Regression

Nothing so far is groundbreaking, or particularly useful. All we have done is explained a way of generalizing the multivariate Normal, but haven't talked about how it can be used in real life. However, you could imagine that starting with a prior over functions, we can form a posterior, $p(f \mid X, y)$ by conditioning on our data. Intiutively, doing this excludes all functions that don't "pass through" our data, $(X, y)$. 

I'll approach Gaussian process regression from a slightly different perspective in this section, building up from Bayesian linear regression. This is a cool approach I found in David MacKay's [book](http://www.inference.org.uk/mackay/itila/book.html), that I haven't seem much elsewhere.

To set the stage, we are interested in modelling a function, $f$, for which we have data, $(X, y)$. We start with a [feature map](https://xavierbourretsicotte.github.io/Kernel_feature_map.html) for the input, $R = \phi(X)$, so that $R$ an $N \times D$ matrix. Then $y = Rw$ and we can assign priors, $p(w)$, to build a posterior distribution for the weights, $p(w \mid y, X)$. This posterior is used to make future predictions and recreate $f = y + \epsilon$. 

$$ p(w \mid y_N, X_N) = \frac{p(y_N \mid X_N, w) p(w)}{p(y_N \mid X_N)} $$

However, in some cases we're only interested in making predictions, and in a Bayesian setting this boils down to 2 distributions: (1) the posterior predictive distribution in order to actually make a prediction and (2) the marginal likelihood for model comparison. 

$$ \text{Posterior predictive: } p(y_{n+1} \mid y_N, X_N) $$

$$ \text{Marginal likelihood: } p(y_{N} \mid X_N) $$

Expanding the formulations from Bayesian linear regression: 

$$ y = Rw \qquad \qquad \text{where: } w \sim N(0, \sigma_w^2) $$

And since $y$ is a linear function of $w$ (which is a Normally distributed random variable), its prior is:

$$ y \sim N(0, \sigma_w^2 RR^T) $$

Accounting for noise in our observations, $\sigma^2_{err}$ the prior on our function, $f$, is: 

$$ f \sim N(0, \sigma_w^2 RR^T + \sigma^2_{err} I) $$

This is how the Gaussian process is a prior over functions. The kernel described in that section is exacly $RR^T = \phi(X)\phi(X)^T$ in this section. Now we can start to create the posterior predictive distribution and marginal likelihood. 

Before we get to the practical stuff, a note about kernels. There are many ways to get confused when first learning about kernels. What helped me is first understanding that a kernel is just a function that accepts 2 inputs and returns how "close" the inputs are to each other. From there, you can go in any direction exploring them, some good articles are: [here](https://www.cs.toronto.edu/~duvenaud/cookbook/), [here](http://mlg.eng.cam.ac.uk/tutorials/06/es.pdf) and [here](https://statisticaloddsandends.wordpress.com/2019/06/28/common-covariance-classes-for-gaussian-processes/).

&nbsp;

#### Simulation Problem

In the first couple sentences of the last section I mentioned that we can condition the GP prior on the observed data to get a posterior distribution. All the observed data will then pass through this posterior distribution over functions. This section will use GP's to extrapolate a simulated function. We don't account for noisy observations, which is of course a terrible assumption in the real world.

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

<!-- <p align="center">
  <img src="/assets/exactGP_simulated_function.png" width="70%" height="70%">
</p> -->

<div class='figure' align="center">
    <img src="/assets/exactGP_simulated_function.png" width="65%" height="65%">
    <div class='caption' width="70%" height="70%">
        <!-- <span class='caption-label'>Figure 1.</span>  -->
        <p> Simulated function we are interested in modelling with a GP. We will take samples from the red section and see how well the GP can recreate the blue section </p>
    </div>
</div>

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

<!-- <p align="center">
  <img src="/assets/squared_exp_kernel_posterior.png" width="100%" height="70%">
</p> -->

<div class='figure' align="center">
    <img src="/assets/squared_exp_kernel_posterior.png" width="90%" height="90%">
    <div class='caption'>
        <p> Posterior distribution after fitting the data in red. The graph on the left shows the confidence interval for the test set (blue region). As we get further and further away from the observed data, the confidence band grows. The graph on the right shows samples from the posterior distrubtion. Because we condition on the data and don't add noise, we are forcing the posterior to "pass through" every single one of our observed datapoints.  </p>
    </div>
</div>





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