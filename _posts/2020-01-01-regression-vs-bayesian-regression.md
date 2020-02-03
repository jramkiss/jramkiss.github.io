---
layout: post
title: "Regression VS Bayesian Regression"
date: 2019-12-12 19:23
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: A motivating example on the power of Bayesian regression over simple linear regression.
---

## Key Differences
- Bayesian models provide uncertainty estimates, which are important in determining how our model performs (how robust our model is) under certain parameter values.
- Under a Bayesian framework, we can encode knowledge about parameters to supplement the model. For example, consider this toy problem: we are trying to find the error in a piece of apparatus that measures the acceleration of objects. We gather data by measuring dropping objects from a height and measuring their acceleration - which should be close to gravity. This "knowledge" about what the acceleration should be can be encoded into a Bayesian model, but cannot be used in a frequentist model.


## Motivating Problem

To apply both regression methods to a real world problem, we'll try to determine the impact of terrain geography on economic growth for nations in Africa and outside of Africa.

This has been studied [here](https://diegopuga.org/papers/rugged.pdf).

```python
DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")

# we'll only use these features:
data = data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)] # remove NaNs
df["rgdppc_2000"] = np.log(df["rgdppc_2000"]) # log real GPD per capita
```

![](regression_VS_bayesian_regression_files/output_2_0.png)


### Simple Linear Regression
#### Model

$$ y = X\beta + \epsilon $$

$$ \epsilon \sim N(0, \sigma^{2}) $$

### Bayesian Regression

To make the model Bayesian, we have to put priors on the parameters, $\beta$ and $\sigma$.


In Bayesian regression, the aim is to quantify uncertainty in our model for different values of our parameters. We do this by learning distributions of the parameters instead of point estimates.
We start by specifying priors for the parameters, and a likelihood for the data

$posterior \propto priors * likelihood$

## Todo
- Write down formulations in a simple way.
- Mention expressiveness of Bayesian model and lack of expressiveness of the frequentist model.
