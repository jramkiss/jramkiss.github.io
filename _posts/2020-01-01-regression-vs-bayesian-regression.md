---
layout: post
title: "Regression VS Bayesian Regression"
date: 2019-12-12 19:23
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: A motivating example on the power of Bayesian regression over simple linear regression.
---

The notebook containing all code and plots for this post can be viewed [here](https://nbviewer.jupyter.org/github/jramkiss/jramkiss.github.io/blob/master/_posts/notebooks/regression_VS_bayesian_regression.ipynb).

## Key Differences
- Bayesian models provide uncertainty estimates, which are important in determining how our model performs (how robust our model is) under certain parameter values.
- Under a Bayesian framework, we can encode knowledge about parameters to supplement the model. For example, consider this toy problem: we are trying to find the error in a piece of apparatus that measures the acceleration of objects. We gather data by measuring dropping objects from a height and measuring their acceleration - which should be close to gravity. This "knowledge" about what the acceleration should be can be encoded into a Bayesian model, but cannot be used in a frequentist model.


## Motivating Problem

To apply both regression methods to a real world problem, we'll try to determine the impact of terrain geography on economic growth for nations in Africa and outside of Africa.
The predictors in this case are: `rugged` - denoting the geography of a country, `cont_africa` - denoting whether or not a country is in Africa and `cont_africa_x_rugged` - an interaction term to help the model. The response is `rgdppc_2000` - log real GDP per capita.

This has been studied [here](https://diegopuga.org/papers/rugged.pdf).

```python
DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")

df = data[["cont_africa", "rugged", "rgdppc_2000"]] # we only need these features
df = df[np.isfinite(df.rgdppc_2000)] # remove NaNs

# real GPD per capita is skewed, so we'll log it
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

# adding feature to capture the interaction between "cont_africa" and "rugged"
# this will be important for comparing the slopes at the end
df["cont_africa_x_rugged"] = df["cont_africa"] * df["rugged"]
```


![](/assets/africa_data_viz.png)
<!--![Figure1](/assets/word2vec_viz.png)-->


### Ordinary Linear Regression

In the model below, $X$ is our data and $y$ is the response. Ordinary linear regression uses maximum likelihood to recover _point estimates_ for model parameters $(\beta, \sigma)$. What this means is that in the end, our model is summarized by a handful of numbers, each an estimate of a parameter.

$$
\begin{equation}
y = \beta_0 + X_1\beta_1 + X_2\beta_2 + X_3\beta_3 + \epsilon
\tag{1}
\end{equation}
$$

$$ \beta = (\beta_0, \beta_1, \beta_2, \beta_3) $$

$$ \epsilon \sim N(0, \sigma^{2}) $$

Here's the code for fitting a linear regression model in Python using `sklearn`.

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

![](/assets/linear_regression_fit.png)

Judging from the regression lines, there's definitely a difference in the effect between African and non-African countries. We can calculate the gradients of each of these slopes and compare them.

```python
# backout the slopes of lines for nations in and out of Africa
print("Slope for African nations: ", coef["rugged"] + coef["cont_africa_x_rugged"])
print("Slope for non-African nations: ", coef["rugged"])
```

Are we confident in these numbers? What if the model didn't have enough data and its confidence in these parameters estimates was very low? This is where Bayesian methods shine.

### Bayesian Regression


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

We'll use [Pyro](http://pyro.ai) to write the Bayesian model. Since the point of this post is to compare Bayesian regression to Ordinary linear regression, I'll be using Pyro as a tool and will skip over detailed explanation of the code. Luckily Pyro has amazing examples on their docs, and I have a link to my notebook [here](https://nbviewer.jupyter.org/github/jramkiss/jramkiss.github.io/blob/master/_posts/notebooks/regression_VS_bayesian_regression.ipynb).

We start by building a class, `BayesianRegression`, that will specify the regression model and parameter priors when initialized. The `forward` method is used to generate values for the response based on samples from the priors.

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

For posterior inference, we'll use stochastic variational inference, which approximates the posterior distribution by minimizing ELBO loss (evidence lower bound). The `guide` code below is Pyro's way of allowing us to specify a distribution to model the posterior after, we'll bypass specifying this outselves and use the `AutoDiagonalNormal` function, which does this automatically for us.

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
