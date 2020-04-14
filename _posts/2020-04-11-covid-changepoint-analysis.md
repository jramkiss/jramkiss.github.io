---
layout: post
title: "Detecting Changes in COVID-19 Cases with Bayesian Models"
date: 2020-04-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Usinng Bayesian change point analysis with regression to determine when COVID-19 cases started to change in different countries.
---

## Problem

There's an amazing example of Bayesian change point analysis in the book [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_TFP.ipynb), and

With the current global pandemic and its associated resources (data, analyses, etc.), I've been trying for some time to come up with an interesting COVID-19 problem to attack with statistics. After looking at the number of confirmed cases for some counties, it was clear to me that at _some_ date, the number of new cases stopped being exponential and its distribution changed. However, this date was different for each country (obviously). I propose a Bayesian model for estimating the date that the number of new confirmed cases in a particular country.


## Model

We want to describe $y$, log of the number of new confirmed cases each day, which we'll do using a segmented regression model. The point at which we segment will be determined by a learned parameter, $\tau$. The model is below:

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

**Posterior:**

$$
\begin{equation*}
p(w, b, \tau, \sigma \mid y) = p(y \mid w, b, \tau, \sigma) \quad p(w, b \mid \tau) \quad p(\tau) \quad p(\sigma)
\end{equation*}
$$


In other words, we model $y$ as $w_1t + b_1$ for days up until day $\tau$. After that we model $y$ as $w_2t + b_2$.

&nbsp;

**Prior Specification**

Virus growth is sensitive to population dynamics of individual countries and we are limited in the amount of data available, so it is important to supplement the model with appropriate priors. For the prior means of the bias terms, we use the mean of the first and forth quartiles of $y$ respectively.

$$
b_1 \sim N(\mu_{q_1}, 1) \qquad \qquad b_2 \sim N(\mu_{q_4}, 1)
$$


We also know that the growth will be positive in the beginning of the model, so we can put a reasonably strong prior on $w_1$. Assuming that we want the majority of values to lie between $(0, 1)$, an appropriate prior can be $w_1 \sim N(0.5, 0.25)$.

I'm hesitant to use the same logic for $w_2$, as the model should be flexible enough to capture gradients similar to $w_1$ in the case where there is no real change in the data. We'll see examples about this in the Results section by testing the model on data up to a particular date. For now, we want the prior for $w_2$ to be symmetric about $0$, with the majority of values lying between $(-0.5, 0.5)$. We'll use $w_2 \sim N(0, 0.25)$.

As for $\tau$, since at this time we don't have access to all the data (the virus is ongoing), we're unable to have a completely flat prior and have the model estimate it. Instead, the assumption is made that the change is more likely to occur in the second half of the date range at hand, so we use $\tau \sim Beta(4, 3)$.

&nbsp;

```python
class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features, b1_mu, b2_mu):
        super().__init__()
        self.linear1 = PyroModule[nn.Linear](in_features, out_features, bias = False)
        self.linear1.weight = PyroSample(dist.Normal(0, 1.).expand([1, 1]).to_event(1))
        self.linear1.bias = PyroSample(dist.Normal(b1_mu, 1.))

        # could possibly have stronger priors for the 2nd regression line, because we wont have as much data
        self.linear2 = PyroModule[nn.Linear](in_features, out_features, bias = False)
        self.linear2.weight = PyroSample(dist.Normal(0., 1.).expand([1, 1]).to_event(1))
        self.linear2.bias = PyroSample(dist.Normal(b2_mu, 2.))

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

## Results

### Data and Processing

The data used was downloaded from [Kaggle](https://www.kaggle.com/imdevskp/corona-virus-report). Available to us is the number of daily confirmed cases in each country, and Figure 1 shows this data in Italy. It is clear that there are some inconsistencies in how the data is reported, for example, there are no new confirmed cases on March 12th, but nearly double the expected (based solely on intuition) cases on March 13th. In cases like this, the data was split between the two days.

<!-- figure 1: daily confirmed cases in Italy -->
![](/assets/italy-daily-cases.png)

### Canada

Since I live in Canada and have exposure to the dates precautions started, I'll start here.

**Prior**

$$
w_1, w_2 \sim N(0, 0.5) \qquad b_1 \sim N(1.1, 1) \qquad b_2 \sim N(7.2, 1)
$$

Posterior plots for Canada

<!-- figure 1: daily confirmed cases in Italy -->
![](/assets/canada-posterior-plots.png)

<br/>

Change date for Canada: 2020-03-29

Regression plot for Canada:

![](/assets/canada-regression-plot.png)

### Canada with Less Data

To test the model's robustness to countries that have not began to flatten the curve yet, we'll look at data from Canada up until March 29th. This is the day that the model estimated curve flattening began.

**Prior**

$$
w_1, w_2 \sim N(0, 0.5) \qquad b_1 \sim N(0.9, 1) \qquad b_2 \sim N(6.4, 1)
$$

The posteriors for $w_1$ and $w_2$ overlap, and the posterior for $\tau$ is bi-model. This is a good sign, because it shows that the model is trying to estimate an appropriate $\tau$ but cannot because it doesn't exist.

![](/assets/canada-march29-posterior-plots.png)

<img src="/assets/canada-march29-posterior-plots.png" alt="drawing" width=100%/>


Even though an appropriate $\tau$ doesn't exist, we're still able to describe the data well, as shown by samples from the likelihood distribution below.

![](/assets/canada-march29-regression-plot.png)
