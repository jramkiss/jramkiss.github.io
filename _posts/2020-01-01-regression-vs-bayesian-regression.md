---
layout: post
title: "Ordinary VS Bayesian Linear Regression"
date: 2019-12-12 19:23
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: A walkthrough of the intuition behind Bayesian regression and a practical comparison to ordinary linear regression.
---

Bayesian methods are usually shrouded in mystery, draped behind walls of math and stats that no practitioner has the patience to understand. Why would I even use this complicated black magic if a neural network is better? Also, since when is there a Bayesian version of good ole linear regression?? And while we're at it, what in the world is [MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) and should I even care?
All these questions will be answered in this blog post. We'll fit an ordinary linear regression and a Bayesian linear regression model to a toy problem, and walk through the intuition behind Bayesian thinking. The post itself isn't code-heavy, but rather provides little snippets for you to follow along. I've included the notebook with all the code [here](https://nbviewer.jupyter.org/github/jramkiss/jramkiss.github.io/blob/master/_posts/notebooks/regression_VS_bayesian_regression.ipynb).

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

coef = dict([i for i in zip(list(x.columns), reg.coef_)]); coef
print("Intercept: %f" % reg.intercept_)
print("Coefficient of Determination: %f" % reg.score(x, y))
```

&nbsp;

![](/assets/linear_regression_fit.png)

&nbsp;

Judging from the regression lines, there's definitely a difference in the effect between African and non-African countries. We can calculate the gradients of each of these slopes and compare them.

&nbsp;

```python
# backout the slopes of lines for nations in and out of Africa
print("Slope for African nations: ", coef["rugged"] + coef["cont_africa_x_rugged"])
print("Slope for non-African nations: ", coef["rugged"])
```

&nbsp;

Are we confident in these numbers? What if the model didn't have enough data and its confidence in these parameters estimates was very low? This is where Bayesian methods shine.

&nbsp;

## Bayesian Regression



To make the ordinary linear regression model Bayesian, all we really have to do is specify priors for the parameters, $(\beta$, $\sigma$). However, to capture the essence of Bayesian methodology, let's think of the problem in a completely different way.

We have a model for our data (1), that is based on observations $(X, y)$ and parameters $(\beta, \sigma)$. Because $\epsilon$ is Normally distributed, $y$ is also Normally distributed under this model. So we can write down a distribution for $y$ and interpret it as the probability that the data came from our model.

$$
\begin{equation}
p(y | \beta, \sigma) \sim N (X\beta, \sigma^2)
\tag{2}
\end{equation}
$$

We're interested in estimating values for $\beta$ so that we can plug them back into our model. Before we get to estimating, the Bayesian framework allows us to add anything we know about our parameters to the model. In this case we don't really know anything about $\beta$... which is fine, but we do know that $\sigma$ can't be less than 0 because it is a standard deviation. The encoding of this knowledge before we start estimation is referred to as _prior specification_.

Since we don't know much about $\beta$, we'll use an uninformative (flat) prior. For $\sigma$ we'll use $U(0, 10)$, which ensures positive values.

$$ p(\beta) \sim N(0, 5) $$

$$ p(\sigma) \sim U(0, 10) $$

Now we want to get the distribution $p(\beta  y, \sigma)$, which is proportional to the likelihood (2) times the priors. This is called the posterior formulation, and it is usually intractable (cannot be written down). Here's where MCMC and variational inference come into play with Bayesian methods - they are used to draw samples from the posterior.

&nbsp;

We'll use [Pyro](http://pyro.ai) to write the Bayesian model. Since the point of this post is to compare Bayesian regression to Ordinary linear regression, I'll be using Pyro as a tool and will skip over detailed explanation of the code. Luckily Pyro has amazing examples on their docs, and I have a link to my notebook [here](https://nbviewer.jupyter.org/github/jramkiss/jramkiss.github.io/blob/master/_posts/notebooks/regression_VS_bayesian_regression.ipynb).

We start by building a class, `BayesianRegression`, that will specify the regression model and parameter priors when initialized. The `forward` method is used to generate values for the response based on samples from the priors.

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

For posterior inference, we'll use stochastic variational inference, which approximates the posterior distribution by minimizing ELBO loss (evidence lower bound). The `guide` code below is Pyro's way of allowing us to specify a distribution to model the posterior after, we'll bypass specifying this outselves and use the `AutoDiagonalNormal` function, which does this automatically for us.

&nbsp;

```python
model = BayesianRegression(3, 1)
auto_guide = AutoDiagonalNormal(model)

svi = SVI(model = model, # bayesian regression class
          guide = auto_guide, # using auto guide
          optim = pyro.optim.Adam({"lr": 0.05}),
          loss=Trace_ELBO())
```

Now we can run the inference loop:

```python
num_iterations = 2500
# param_store is where Pyro stores all learned parameters of the model
pyro.clear_param_store()
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(x_data, y_data)
    if j % 250 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))
```

Now that we've ran the inference loop, we can look at the learned parameters by iterating over the items in Pyro's _param_store_.

```python
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))
```

        AutoDiagonalNormal()
        AutoDiagonalNormal.loc Parameter containing:
        tensor([-2.2693, -1.8370, -0.1964,  0.3043,  9.1820])
        AutoDiagonalNormal.scale tensor([0.0615, 0.1746, 0.0426, 0.0829, 0.0771])


## Key Differences
- Bayesian models provide uncertainty estimates, which are important in determining how our model performs (how robust our model is) under certain parameter values.
- Under a Bayesian framework, we can encode knowledge about parameters to supplement the model. For example, consider this toy problem: we are trying to find the error in a piece of apparatus that measures the acceleration of objects. We gather data by measuring dropping objects from a height and measuring their acceleration - which should be close to gravity. This "knowledge" about what the acceleration should be can be encoded into a Bayesian model, but cannot be used in a frequentist model.


### References
- Study about terrain and economic growth [here](https://diegopuga.org/papers/rugged.pdf).
