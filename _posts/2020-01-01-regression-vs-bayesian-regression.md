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

Now we want to get the distribution $ p(\beta | y, \sigma) $, which is proportional to the likelihood (2) times the priors. This is called the posterior formulation, and it is usually intractable (cannot be written down). Here's where MCMC and variational inference come into play with Bayesian methods - they are used to draw samples from the posterior.

We'll use [Pyro](http://pyro.ai) for the geography and GDP problem. Pyro offers numerous ways of doing posterior inference.



## Todo
- Write down formulations in a simple way.
- Mention expressiveness of Bayesian model and lack of expressiveness of the frequentist model.
