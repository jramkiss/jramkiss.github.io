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


### Simple Linear Regression
#### Model

$$ y = X\beta + \epsilon $$

$$ \epsilon \sim N(0, \sigma^{2}) $$

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


<!-- plots for regression fit -->


```python
# backout the slopes of lines for nations in and out of Africa
print("Slope for African nations: ", coef["rugged"] + coef["cont_africa_x_rugged"])
print("Slope for non-African nations: ", coef["rugged"])
```

### Bayesian Regression

To make the model Bayesian, we have to put priors on the parameters, $\beta$ and $\sigma$.


In Bayesian regression, the aim is to quantify uncertainty in our model for different values of our parameters. We do this by learning distributions of the parameters instead of point estimates.
We start by specifying priors for the parameters, and a likelihood for the data

$posterior \propto priors * likelihood$

## Todo
- Write down formulations in a simple way.
- Mention expressiveness of Bayesian model and lack of expressiveness of the frequentist model.


## Appendix

### Simple Linear Regression Example

#### Plotting Data
```python
african_nations = df[df["cont_africa"] == 1]
non_african_nations = df[df["cont_africa"] == 0]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
ax[0].scatter(non_african_nations["rugged"],
            non_african_nations["rgdppc_2000"])
ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="Non African Nations")
ax[1].scatter(african_nations["rugged"],
                african_nations["rgdppc_2000"])
ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="African Nations");
```

#### Plots for Regression Fit
```python
african_nations = df[df["cont_africa"] == 1]
non_african_nations = df[df["cont_africa"] == 0]

# predict GPD from continent and terrain
african_gdp = reg.predict(african_nations[features])
non_african_gdp = reg.predict(non_african_nations[features])

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), sharey=True)
ax[0].plot(non_african_nations["rugged"],
           non_african_gdp,
           color = "orange")
ax[0].scatter(non_african_nations["rugged"],
              non_african_nations["rgdppc_2000"])
ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="predicted log GDP (2000)",
          title="Non African Nations")

ax[1].plot(african_nations["rugged"],
           african_gdp,
           color = "orange")
ax[1].scatter(african_nations["rugged"],
              african_nations["rgdppc_2000"],
              color = "red")
ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="predicted log GDP (2000)",
          title="African Nations");

ax[2].scatter(df["rugged"],
              reg.predict(df[features]),
              color = "green")
ax[2].scatter(african_nations["rugged"],
              african_nations["rgdppc_2000"],
              color = "red")
ax[2].scatter(non_african_nations["rugged"],
              non_african_nations["rgdppc_2000"])
ax[2].set(xlabel="Terrain Ruggedness Index",
          ylabel="predicted log GDP (2000)",
          title="All Nations");
```
