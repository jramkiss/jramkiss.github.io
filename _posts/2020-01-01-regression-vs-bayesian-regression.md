---
layout: post
title: "Ordinary VS Bayesian Linear Regression"
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: A walkthrough of the intuition behind Bayesian regression and a practical comparison to ordinary linear regression.
---

Bayesian methods are usually shrouded in mystery, draped behind walls of math and stats that no practitioner has the patience to understand. Why would I even use this complicated black magic if a neural network is better? Also, since when is there a Bayesian version of simple linear regression?? And while we're at it, what in the world is [MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) and should I even care?
The goal of this post is to answer all these questions and to explain the intuition behind Bayesian thinking without using math. To do this, we'll fit an ordinary linear regression and a Bayesian linear regression model to a practical problem.

The post itself isn't code-heavy, but rather provides little snippets for you to follow along. I've included the notebook with all the code [here](https://nbviewer.jupyter.org/github/jramkiss/jramkiss.github.io/blob/master/_posts/notebooks/regression_VS_bayesian_regression.ipynb).

&nbsp;

## Problem

**Is the relationship between terrain ruggedness and economix growth the same for countries inside and outside of Africa?**

Our data is as follows:

- `rugged`: Ruggedness of a country's terrain
- `cont_africa`: Whether or not a country is in Africa
- `rgdppc_2000` - Real GDP per capita


In our models, the response will be `rgdppc_2000` and the predictors are: `rugged`, `cont_africa` and an interaction term between these two, `cont_africa_x_rugged`. This interaction term helps the model a lot, I encourage you to run the code and think about the model parameters if you'd like to find out how.
We can use the slope of regression lines for countries inside and outside Africa to determine the relationship between terrain ruggedness and GDP.

Here's what the data looks like.

&nbsp;

![](/assets/africa_data_viz.png)
<!--![Figure1](/assets/word2vec_viz.png)-->

&nbsp;

## Ordinary Linear Regression

In (1), $X$ is the [data](#problem) and $y$ is the response, `rgdppc_2000`. The parameters are $(\beta, \sigma)$. Ordinary linear regression uses maximum likelihood to find estimates for the parameters, then we can use these estimates to compare the slopes.

&nbsp;

$$
\begin{equation}
y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_3X_3 + \epsilon
\tag{1}
\end{equation}
$$

$$ \beta = (\beta_0, \beta_1, \beta_2, \beta_3) $$

$$ \epsilon \sim N(0, \sigma^{2}) $$

&nbsp;

```python
features = ["rugged", "cont_africa_x_rugged", "cont_africa"]
x = df[features]
y = df["rgdppc_2000"]

reg = LinearRegression()
_ = reg.fit(x, y)
# save coefficients
coef = dict([i for i in zip(list(x.columns), reg.coef_)])
```

Now we can plot the regression lines for African and Non-African nations. Judging from these lines, there's definitely a difference in relationship - at the very least, the two gradients are of opposite signs.

&nbsp;

![](/assets/linear_regression_fit.png)

&nbsp;

```python
# backout the slopes of lines for nations in and out of Africa
print("Slope for African nations: ", coef["rugged"] + coef["cont_africa_x_rugged"])
print("Slope for non-African nations: ", coef["rugged"])
```

Are we confident in these numbers? What if the model didn't have enough data and its confidence in these parameters estimates was very low? This is where Bayesian methods shine.

&nbsp;

## Bayesian Regression

To make the ordinary linear regression model Bayesian, we have to specify priors for the parameters, $(\beta$, $\sigma$). However before we get there, to capture the essence of Bayesian methodology let's start with the linear model from (1) and build up.

We have a model for our data (1), that is based on observations $(X, y)$ and parameters $(\beta, \sigma)$. Because $\epsilon$ is Normally distributed, $y$ is also Normally distributed in this model. Assuming we have values for $(\beta, \sigma)$, we can write down a distribution for $y$.

$$
\begin{equation}
p(y | \beta, \sigma) \sim N (X\beta, \sigma^2)
\tag{2}
\end{equation}
$$

Remember that we're interested in estimating values for $\beta$ so that we can plug them back into our model and interpret the regression slopes. Before we get to estimating, the Bayesian framework allows us to add anything we know about our parameters to the model. In this case we don't really know anything about $\beta$ which is fine, but we do know that $\sigma$ can't be less than 0 because it is a standard deviation. The encoding of this knowledge before we start estimation is referred to as _prior specification_.

Since we don't know much about $\beta$, we'll use an uninformative (flat) prior and for $\sigma$ we'll use $U(0, 10)$, which ensures only positive values.

$$ p(\beta) \sim N(0, 5) $$

$$ p(\sigma) \sim U(0, 10) $$

Now we want to get the distribution $p(\beta | y, \sigma)$, which is proportional to the likelihood (2) multiplied by the priors. This is called the posterior formulation.
In real world applications, the posterior distribution is usually intractable (cannot be written down). Here's where MCMC and variational inference come into play with Bayesian methods - they are used to draw samples from the posterior so that we can learn about our parameters. At this point you may be wondering why are we concerned with a distribution when $\beta$ a number (vector of numbers)? Well the distribution gives us more information about $\beta$, we can then find _point estimates_ by taking the mean, median or randomly sampling from this distribution.


To write the Bayesian model in Python, we'll use [Pyro](http://pyro.ai). Since the point of this post is to compare Bayesian regression to Ordinary linear regression, I'll be using Pyro as a tool and will skip over detailed explanation of the code. Luckily Pyro has amazing examples in their docs if you want to learn more.

&nbsp;

```python
class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        # specify the linear model
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        # specify priors on `weight` and `bias`
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1) # prediction
        # sample from the likelihood distribution
        obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean
```

&nbsp;

For posterior inference, we'll use stochastic variational inference, which is a method used to approximate the posterior. The `guide` code below is Pyro's way of allowing us to specify a distribution to model the posterior after, we'll bypass specifying this outselves and use the `AutoDiagonalNormal` function, which does this automatically for us.

&nbsp;

```python
model = BayesianRegression(3, 1)
auto_guide = AutoDiagonalNormal(model)

svi = SVI(model = model, # bayesian regression class
          guide = auto_guide, # using auto guide
          optim = pyro.optim.Adam({"lr": 0.05}),
          loss=Trace_ELBO())
```

Now that we've ran the inference loop, we can generate posterior samples. Pyro uses the `Predictive` class to generate posterior samples. Specifying different `return_sites` will tell Pyro which parameters to sample.

Here is where the advantage of Bayesian linear regression starts to show itself. With Ordinary linear regression we'd end up with point estimates of parameters, but now we have a way of seeing how confident the model is in the parameter estimate. We can plot the posterior distributions for each of the parameters to see the difference in confidence.

&nbsp;

```python
num_samples = 1000
predictive = Predictive(model = model,
                        guide = auto_guide,
                        num_samples = num_samples,
                        return_sites=("linear.weight", "linear.bias", "obs", "_RETURN"))
pred = predictive(x_data)
weight = pred["linear.weight"]
weight = weight.reshape(weight.shape[0], 3)
bias = pred["linear.bias"]

# columns of x_data: cont_africa, rugged, cont_africa_x_rugged
print("Mean for posterior distributions: ", torch.mean(weight, 0))
print("97.5 percentile: ", weight.kthvalue(int(num_samples * 0.975), dim = 0)[0]) # find the 97.5 percentile value
print("2.5 percentile: ", weight.kthvalue(int(num_samples * 0.025), dim = 0)[0]) # find the 2.5 percentile value

fig = plt.figure(figsize = (10, 5))
sns.distplot(weight[:, 0], kde_kws = {"label": "`cont_africa` Posterior Samples"})
sns.distplot(weight[:, 1], kde_kws = {"label": "`rugged` Posterior Samples"})
sns.distplot(weight[:, 2], kde_kws = {"label": "`cont_africax_rugged` Posterior Samples"})
sns.distplot(bias[:, 0], kde_kws = {"label": "Bias term Posterior Samples"})

fig.suptitle("Posterior Distributions");
```

    Mean for posterior distributions:  tensor([[-1.9931, -0.1638,  0.2811]])
    97.5 percentile:  tensor([[-1.6826, -0.0764,  0.4010]])
    2.5 percentile:  tensor([[-2.2934, -0.2515,  0.1571]])

&nbsp;

<!-- space for plot of posterior disitbutrions -->
![](/assets/posteriors.png)

&nbsp;


```python
weight = weight.reshape(weight.shape[0], 3)
in_africa = weight[:, 1] + weight[:, 2] # rugged + cont_africa_x_rugged
outside_africa = weight[:, 1] # rugged

fig = plt.figure(figsize=(10, 6))
sns.distplot(in_africa,
             kde_kws = {"label": "African nations"},)
sns.distplot(outside_africa,
             kde_kws={"label": "Non-African nations"})
fig.suptitle("Density of Slope : log(GDP) vs. Terrain Ruggedness");
```

![](/assets/bayesian_slopes.png)
<!-- space for plot of difference in slopes -->

&nbsp;

Getting back to the questions we asked at the beginning of this post:

- Why would I even use this complicated black magic if a neural network is better?: Different tools for different jobs. Neural networks are not as expressive as Bayesian methods. If all we care about is predictive power, then there's little need for parameter confidence intervals and a non-Bayesian approach will suffice in most instances. However, when we want to do inference and compare effects (coefficients) with some level of confidence, Bayesian methods shine.
- Since when is there a Bayesian version of simple linear regression?: There's a Bayesian version of most things. If we have a model for data that can be expressed as a probability distribution, then we can specify distributions for its parameters and come up with a Bayesian formulation.
- What in the world is [MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) and should I even care?: MCMC is a family of methods used to sample distributions we can't write down (you don't need to care about different types of MCMC algorithms). However you should know that in Bayesian problems, the posterior distribution is not usually well defined, so we use MCMC algorithms to sample these undefined posteriors. There are other ways to sample / approximate distributions, such as variational inference.



### Resources
- Study about terrain and economic growth [here](https://diegopuga.org/papers/rugged.pdf).
- Pyro's [tutorial](http://pyro.ai/examples/bayesian_regression.html) on Bayesian linear regression
