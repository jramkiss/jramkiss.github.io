---
layout: post
title: "Detecting Changes in COVID-19 Cases with Bayesian Models"
date: 2020-04-11 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Bayesian model to estimate the date that flattening of new COVID-19 cases started.
---

## Problem

With the current global pandemic and its associated resources (data, analyses, etc.), I've been trying for some time to come up with an interesting COVID-19 problem to attack with statistics. After looking at the number of confirmed cases for some counties, it was clear that at _some_ date, the number of new cases stopped being exponential and its distribution changed. However, this date was different for each country (obviously). This post introduces and discusses a Bayesian model for estimating the date that the distribution of new COVID-19 cases in a particular country changes.


## Model

We want to describe $y$, log of the number of new cases each day, which we'll do using a segmented regression model. The point at which we segment will be determined by a learned parameter, $\tau$. This is outlined below:

**Likelihood:**

$$
\begin{equation*}
  \begin{split}
    y = wt + b + \epsilon
  \end{split}
  \text{, } \qquad \qquad
  \begin{split}
    \epsilon \sim N(0, \sigma^2) \\[10pt]
    p(y \mid w, b, \sigma) \sim N(wt, \sigma^2)
  \end{split}
  \\[15pt]
\end{equation*}
$$

$$
\begin{equation*}
\begin{split} \text{Where: } \qquad \qquad \end{split}
\begin{split}
w &= \begin{cases}
  w_1 & \text{if } \tau \le t\\
  w_2 & \text{if } \tau \gt t\\
\end{cases} \\
b &= \begin{cases}
  b_1 & \text{if } \tau \le t\\
  b_2 & \text{if } \tau \gt t\\
\end{cases}
\end{split}
\\[10pt]
\end{equation*}
$$

**Priors:**

$$
\begin{equation*}
  w_1 \sim N(\mu_{w_1}, \sigma_{w_1}^2) \qquad \qquad
  w_2 \sim N(\mu_{w_2}, \sigma_{w_2}^2)
  \\[10pt]
  b_1 \sim N(\mu_{b_1}, \sigma_{b_1}^2) \qquad \qquad
  b_2 \sim N(\mu_{b_2}, \sigma_{b_2}^2)
  \\[10pt]
  \tau \sim Beta(\alpha, \beta) \qquad \qquad
  \sigma \sim U(0, 2)
\end{equation*}
$$

&nbsp;

In other words, $y$ will be modeled as $w_1t + b_1$ for days up until day $\tau$. After that it will be modeled as $w_2t + b_2$.

&nbsp;

### Prior Interpretation and Specification

Virus growth is sensitive to population dynamics of individual countries and we are limited in the amount of data available, so it is important to supplement the model with appropriate priors.

Starting with $w_1$ and $w_2$, these parameters can be interpreted as the growth rate of the virus before and after the date change. We know that the growth will be positive in the beginning, so we can put a reasonably strong prior on $w_1$. Assuming that we want the majority of values to lie between $(0, 1)$, $w_1 \sim N(0.5, 0.25)$ will be used as the prior.
We'll use similar logic for $p(w_2)$, but will have to keep in mind flexibility. Without a flexible enough prior here, the model won't do well in cases where there is no real change point in the data. In these cases, $w_2 \approx w_1$, and we'll see and example of this in the [Results](#results) section. For now, we want $p(w_2)$ to be symmetric about $0$, with the majority of values lying between $(-0.5, 0.5)$. We'll use $w_2 \sim N(0, 0.25)$.

Next are the bias terms, $b_1$ and $b_2$. Priors for these parameters are especially sensitive to country characteristics. Countries that are more exposed to COVID-19 (for whatever reason), will have more confirmed cases at its peak than countries that are less exposed. This will directly affect the posterior distribution for $b_2$. In order to adapt this parameter to different countries, the mean of the first and forth quartiles of $y$ are used as $mu_{b_1}$ and $mu_{b_2}$ respectively. The standard deviation for these priors is taken as half the mean value in order to preserve flexibility.

$$
b_1 \sim N(\mu_{q_1}, \frac{\mu_{q_1}}{2}) \qquad \qquad b_2 \sim N(\mu_{q_4}, \frac{\mu_{q_4}}{2})
$$

As for $\tau$, since at this time we don't have access to all the data (the virus is ongoing), we're unable to have a completely flat prior and have the model estimate it. Instead, the assumption is made that the change is more likely to occur in the second half of the date range at hand, so we use $\tau \sim Beta(4, 2)$.

&nbsp;

```python
class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features, b1_mu, b2_mu):
        super().__init__()
        self.linear1 = PyroModule[nn.Linear](in_features, out_features, bias = False)
        self.linear1.weight = PyroSample(dist.Normal(0, 1.).expand([1, 1]).to_event(1))
        self.linear1.bias = PyroSample(dist.Normal(b1_mu, 1_mu/2))

        # could possibly have stronger priors for the 2nd regression line, because we wont have as much data
        self.linear2 = PyroModule[nn.Linear](in_features, out_features, bias = False)
        self.linear2.weight = PyroSample(dist.Normal(0., 1.).expand([1, 1]).to_event(1))
        self.linear2.bias = PyroSample(dist.Normal(b2_mu, b2_mu/2))

    def forward(self, x, y=None):
        tau = pyro.sample("tau", dist.Beta(4, 2))
        sigma = pyro.sample("sigma", dist.Uniform(0., 2.))
        # fit lm's to data based on tau
        sep = int(np.ceil(tau.detach().numpy() * len(x)))
        mean1 = self.linear1(x[:sep]).squeeze(-1)
        mean2 = self.linear2(x[sep:]).squeeze(-1)
        mean = torch.cat((mean1, mean2))
        # sample from the posterior
        obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean
```
&nbsp;


## Data and Inference

The data used was downloaded from [Kaggle](https://www.kaggle.com/imdevskp/corona-virus-report). Available to us is the number of daily confirmed cases in each country, and Figure 1 shows this data in Italy. It is clear that there are some inconsistencies in how the data is reported, for example, there are no new confirmed cases on March 12th, but nearly double the expected cases on March 13th. In cases like this, the data was split between the two days.

The virus also starts at different times in different countries. Because we have a regression model, it is inappropriate to include data prior to the virus being in a particular country. This date is chosen by hand for each country based on the progression of new cases and is never the date the first patient is recorded. The "start" date is better interpreted as the date the virus started to consistently grow, as opposed to the date the patient 0 was recorded.

&nbsp;

<!-- figure 1: daily confirmed cases in Italy -->
![](/assets/italy-daily-cases.png)

&nbsp;

Hamiltonian Monte Carlo is used for posterior sampling.

```python
model = BayesianRegression(1, 1,
                           b1_mu = bias_1_mean,
                           b2_mu = bias_2_mean)
# mcmc
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel,
            num_samples=300,
            warmup_steps = 100,
            num_chains = 4)
mcmc.run(x_data, y_data)
samples = mcmc.get_samples()
```

&nbsp;

## Results

### Canada

Since I live in Canada and have exposure to the dates precautions started, modeling will start here. We'll use February 27th as the date the virus "started".

**Prior**

$$
w_1, w_2 \sim N(0, 0.5) \qquad b_1 \sim N(1.1, 0.5) \qquad b_2 \sim N(7.2, 3.6)
$$

&nbsp;

**Posterior Distributions**

<!-- figure 1: daily confirmed cases in Italy -->
![](/assets/canada-posterior-plots.png)

&nbsp;

Starting the the posteriors for $w_1$ and $w_2$, if there was no change in the data we would expect to see these two distributions close to each other as they govern the growth rate of the virus. It is a good sign that these distributions, along with the posteriors for $b_1$ and $b_2$, don't overlap. The posterior for $\tau$ is also symmetric about its mean and doesn't show signs of bi-modality. All of this is evidence that the change point estimated by our model is true.

This change point was estimated as: **2020-03-29**

As a side note, with no hard science attached, my company issued a mandatory work from home policy on March 16th, 13 days before the model's estimated change date. Assuming the incubation period for the virus is up to 14 days as reported, these dates align!
The model fit along with 95% credible interval bands can be seen in the plot below. Also included is the true number of daily cases.

&nbsp;

![](/assets/canada-regression-plot.png)

&nbsp;

To diagnose the MCMC, below are trace plots for each parameter. Each of these have mixed well and are stationary.

![](/assets/canada-trace-plots.png)


### Canada with Less Data

To test the model's robustness to countries that have not began to flatten the curve yet, we'll look at data from Canada up until March 29th. This is the day that the model estimated curve flattening began. Now just because there isn't a true change date doesn't mean the model will output "No change". We'll use the posterior distributions to reason that there is no change in the data.

**Prior**

$$
w_1, w_2 \sim N(0, 0.5) \qquad b_1 \sim N(0.9, 1) \qquad b_2 \sim N(6.4, 1)
$$

&nbsp;

**Posterior Distributions**


![](/assets/canada-march29-posterior-plots.png)

The posteriors for $w_1$ and $w_2$ overlap, and the posterior for $\tau$ is bi-model. This is a good sign, because it shows that the model is trying to estimate an appropriate $\tau$ but cannot because it doesn't exist.

Even though an appropriate $\tau$ doesn't exist, the model priors are flexible enough to allow us to still describe the data well, as shown by the plot below.

&nbsp;

![](/assets/canada-march29-regression-plot.png)
