---
layout: post
title: "Estimating NBA Free Throw Percentage with Hierarchical Models"
date: 2021-01-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Exploring pooling and hierarchical models with Numpyro by estimating the free throw percentage for NBA players
---


In this post I explore 3 different formulations for modeling repeated Bernoulli / binary trial data: complete pooling where all items have the same chance of success, no pooling where each item has in independent chance of success and partial pooling where data across items are shared to estimate parameters. To do this I use an example where I try to estimate the batting average of baseball players, with inference in Numpyro. All the code for this post is available [here](https://www.kaggle.com/jramkiss/pooling-in-hierarchical-models-with-numpyro).

In a repeated Bernoulli / binary trial, our data consists of $n$ units where each unit, $i$, records $y_i$ successes in $K_i$ trials / attempts. It essentially consists of a series of attempts with binary outcomes and is easiest explained with examples:

- Baseball batters: Every pitch faced is a trial and every hit is a success. Each batter is a unit
- Basketball players taking free throws: Every free throw is a trial and everytime they make it is a success. Each player is a unit



## Problem and Data

I'll use [NBA free throw data](https://www.kaggle.com/sebastianmantey/nba-free-throws) to model the free throw percentage of top players in the 2015-2016 season. To have a sort of train/test split, only the first quarter of the season will be used to fit the models and the other $\frac{3}{4}$ will be for testing / other stuff. Here's what the data looks like after some manipulation: 


<p align="center">
  <img src="/assets/NBA-free-throw-data.png" width="60%" height="60%">
</p>

&nbsp;

## Overall Model

The three formulations in this post branch out the same canonical model. We have 15 players, $i = 1...15$, and our goal is to estimate the free throw percentage (chance of success) for each one, $\theta_i$. Our data consists of the number of shots made for player $i$, $y_i$, and the number of attempts for each player, $K_i$. Then the number of free throws made, $y_i$, follows a Binomial distribution: 

$$ p(y_i \mid \theta_i, K_i) = \text{Binomial}(\theta_i, K_i) $$

To help with inference, we transform $\theta$ to a log-odds parameter, $\alpha$. Using $\alpha$ will change the distribution of $y_i$ from a Binomial distribution to a BinomialLogit, but this is just a math formality, the intuition is the same. Outside of inference functions, there's no need to remember the $\alpha$ parameter because it's just a transformation of $\theta$.

$$ \alpha = \text{logit}(\theta) = \text{log}\frac{\theta}{1 - \theta}$$

$$ \theta = \text{InverseLogit}(\alpha) = \text{sigmoid}(\alpha) $$

$$ p(y_i \mid K_i, \alpha_i) = \text{BinomialLogit}(K_i, \alpha) $$

We are interested in estimating $\theta_i = \text{sigmoid}(\alpha_i)$, and our 3 formulations make different assumptions to do this. 

&nbsp;

### Complete Pooling - Same $\theta$ for every player

In the complete pooling formulation, each player has the same chance of success parameter. The advantage of this is that we can aggregate all attempts and all successes for the players to "get" more data. However this is a terrible assumption because we know some players are better at making free throws than others. 

For this model, the likelihood and prior are below, notice that $\theta$ (and by extension, $\alpha$) is not indexed because there is only 1. 

$$ p(y_i \mid \theta, K_i) = \text{Binomial}(\theta, K_i) $$

$$ p(y_i \mid K_i, \alpha) = \text{BinomialLogit}(K_i, \alpha) $$

$$ p(\alpha) = N(1, 1) $$

The prior specification for $\alpha$ can be interpreted as $95\%$ of values falling between $0.26$ and $0.95$ chance of success. 

```python 
def fully_pooled(ft_attempts, ft_makes=None):
    num_players = ft_attempts.shape[0]
    alpha = numpyro.sample("alpha", dist.Normal(1, 1)) # prior on \alpha
    theta = numpyro.deterministic("theta", jax.nn.sigmoid(alpha)) # need to use arviz
    with numpyro.plate("num_players", num_players):
        numpyro.sample("obs", dist.BinomialLogits(total_count = ft_attempts, logits=alpha), 
                       obs=ft_makes)
```

The posterior distribution for $\theta$ is below. Judging from the interval, we would be hard-pressed to find a player with a free throw percentage over $83.5\%$, however $9$ out of the $16$ players analyzed have a free throw percentage higher than $83.5\%$. Aside from the assumptions of this model being completely wrong, it seems like the bi-product is gross underestimation of players' abilities. I guess this is expected since there will also be some overestimation for lower-ability players. 


<p align="center">
  <img src="/assets/NBA-free-throw-fully-pooled-theta.png" width="75%" height="75%">
</p>

&nbsp;

### No Pooling - Independent $\theta_i$ for each player

The no pooling model is the exact opposite of the complete pooling model, where each player has a separate and independent chance of success. The formulation looks similar with a subtle difference, $\theta$ now becomes $\theta_i$ because there is a separate one for each player.

$$ p(y_i \mid \theta_i, K_i) = \text{Binomial}(\theta_i, K_i) $$

$$ p(y_i \mid K_i, \alpha_i) = \text{BinomialLogit}(K_i, \alpha_i) $$

$$ p(\alpha_i) = N(1, 1) $$


```python
def no_pooling (ft_attempts, ft_makes = None):
    num_players = ft_attempts.shape[0]
    with numpyro.plate("players", num_players):
        alpha = numpyro.sample("alpha", dist.Normal(1, 1)) # prior
        assert alpha.shape == (num_players,), "alpha shape wrong" # one alpha for each player
        theta = numpyro.deterministic("theta", jax.nn.sigmoid(alpha))
        return numpyro.sample("obs", dist.BinomialLogits(total_count=ft_attempts, logits=alpha), 
                              obs = ft_makes) # likelihood
```

The posterior distributions for each $\theta$ are below:

<p align="center">
  <img src="/assets/NBA-free-throw-no-pooling-theta(1).png" width="100%" height="100%">
  <img src="/assets/NBA-free-throw-no-pooling-theta(2).png" width="100%" height="100%">
</p>

&nbsp;

It's difficult to evaluate these estimates using only this graph, but one thing we can note is the size of the intervals. Many of them overlap significantly with a free throw percentage of $90\%$ and higher. This is an extremely high percentage, close to getting you on the top [50 all-time list](http://www.iweblists.com/sports/basketball/FreeThrowPercent_s.html). So it seems like the no-pooling formulation overestimates player's abilities. 

Now is a good time to say that I know very little about what a "good" free throw percentage is, meaning that you don't need to be a basketball connoisseur to verify whether these predictions make sense. 

&nbsp;

### Partial Pooling - Use all players to estimate base $\theta$

We ideally want a balance between the two extremes of no-pooling and complete-pooling, and this comes in the form of a partially pooled model. This model has a very subtle but important difference to the `no pooling` model which is in how we generate $\alpha_i$. Instead of sampling $\alpha_i$ directly from $N(1, 1)$, we estimate the mean, $\mu$, and standard deviation, $\sigma$, of $p(\alpha_i)$ using hyper-priors. Here, $\mu$ can be interpreted as the population chance of success. 

$$ p(y_i \mid K_i, \theta_i) = \text{Binomial}(K_i, \theta) $$

$$ p(y_i \mid K_i, \alpha_i) = \text{BinomialLogit}(K_i, \alpha) $$

$$ p(\alpha_i \mid \mu, \sigma) = \text{Normal}(\mu, \sigma) $$

$$ p(\mu) = N(1, 1) $$

$$ p(\sigma) = N(0, 1) $$

&nbsp;

```python
def partial_pooling (ft_attempts, ft_makes = None):
    num_players = ft_attempts.shape[0]
    mu = numpyro.sample("mu", dist.Normal(1, 1))
    sigma = numpyro.sample("sigma", dist.Normal(0, 1))
    with numpyro.plate("players", num_players):
        alpha = numpyro.sample("alpha", dist.Normal(mu, sigma))
        theta = numpyro.deterministic("theta", jax.nn.sigmoid(alpha))
        assert alpha.shape == (num_players, ), "alpha shape wrong"
        return numpyro.sample("y", dist.BinomialLogits(logits = alpha, total_count = ft_attempts), 
                              obs = ft_makes)
```

&nbsp;

The plots below compare the posterior densities for the partial pooled and no-pooled models. The intervals from the partial pooled are narrower and seem better calibrated to what we expect to see in real life. Only three intervals overlap with $90\%$, and based on the players they seem reasonable. 


<p align="center">
  <img src="/assets/NBA-free-throw-partial-pooling-theta(1).png" width="100%" height="100%">
  <img src="/assets/NBA-free-throw-partial-pooling-theta(2).png" width="100%" height="100%">
</p>

&nbsp;

### Where does the difference come from?

The partially pooled and non-pooled models have very similar formulations, but produce very different posterior distributions. The most obvious difference in the formulation the prior on $\alpha_i$. The partial pooling formulation has more flexibility here as both $\mu$ an $\sigma$ are estimated from the data. Below I compare $p(\alpha)$ for the partially pooled and non-pooled models and it seems like the partially pooled prior has more variance than the non-pooled model.

I was also interested to see the impact of flatter priors on the model. However, after increasing the prior variance for the non-pooled model, interval estimates were too wide to be useful, this is because we have such small data on each player. On the other hand, the interval estimates produced by the hierarchical model were very similar to before, this is because the hyper-priors are estimated using population data, which we have more of because of pooling. It turns out that as we collect more and more data, the no-pooling and partially pooled formulations converge to the same solutions. 


<p align="center">
  <img src="/assets/NBA-free-throw-priors.png" width="100%" height="100%">
</p>

&nbsp;

### Checking Model Interpretation 

Above I mentioned that an interpretation of $\mu$ in the hierarchical model is the population chance of success. In our complete pooling formulation, $\theta$ is exactly the population chance of success. Here I compare the posterior distributions of these two parameters to see how similar they really are. Before this is done, $\mu$ needs to be transformed from a log-odds parameters to a probability with $\text{sigmoid}(\mu)$.


<p align="center">
  <img src="/assets/NBA-free-throw-mu-and-sigma.png" width="75%" height="75%">
</p>

&nbsp;

## Resources

- [Hierarchical Partial Pooling for Repeated Binary Trials](https://cran.r-project.org/web/packages/rstanarm/vignettes/pooling.html)
- [Numpyro with Arviz](https://arviz-devs.github.io/arviz/getting_started/CreatingInferenceData.html#from-numpyro)
- [Bayesian hierarchical model for the prediction of football results](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf)


<!-- 
## Resources

- [Modeling Rates/Proportions using Beta Regression](https://cran.r-project.org/web/packages/rstanarm/vignettes/betareg.html)
- [Estimating Generalized Linear Models for Binary and Binomial Data](https://cran.r-project.org/web/packages/rstanarm/vignettes/binomial.html)
- [Estimating Generalized Linear Models for Continuous Data](https://cran.r-project.org/web/packages/rstanarm/vignettes/continuous.html)
- [Estimating Generalized Linear Models for Count Data](https://cran.r-project.org/web/packages/rstanarm/vignettes/count.html)
- [Estimating Generalized (Non-)Linear Models with Group-Specific Terms](https://cran.r-project.org/web/packages/rstanarm/vignettes/glmer.html)
- [Estimating Joint Models for Longitudinal and Time-to-Event Data](https://cran.r-project.org/web/packages/rstanarm/vignettes/jm.html)
- [Estimating Regularized Linear Models](https://cran.r-project.org/web/packages/rstanarm/vignettes/lm.html)
- [Estimating Ordinal Regression Models](https://cran.r-project.org/web/packages/rstanarm/vignettes/polr.html)
- [Estimating ANOVA Models](https://cran.r-project.org/web/packages/rstanarm/vignettes/aov.html) -->


<!-- 

# Outstanding Questions
- How do we actually share data in the hierarchical model? What happen if we dont have \alpha in this model?
- Are hierarchical model overparametrized?


## Key Questions

- What really makes multi-level / hierarchical modelling so much "better"? 
- How is a hierarchical model even better? Is it just the change in posterior? What happens if we use flatter priors?
- What happens to parameter variance when we have hierarchical priors?

### Reading Material

- Statistical Rethinking - Chapter 13
- [Multilevel modelling in PyStan](https://widdowquinn.github.io/Teaching-Stan-Hierarchical-Modelling/07-partial_pooling_intro.html): Jupyter notebook with PyStan example
- [Bayesian Hierarchical Modelling at Scale](https://florianwilhelm.info/2020/10/bayesian_hierarchical_modelling_at_scale/): Post comparing PyMC3 to Pyro and using Pyro for a task with lots of data.
- [Notes on Hierarchical Models](https://vioshyvo.github.io/Bayesian_inference/hierarchical-models.html)
- [CMU Hierarchical Models Intro](http://www.stat.cmu.edu/~brian/463-663/week10/Chapter%2009.pdf)
- [Best of both worlds: Hierarchical models](https://twiecki.io/blog/2014/03/17/bayesian-glms-3/)
- [Radon data analysis](https://github.com/fonnesbeck/multilevel_modeling/blob/master/multilevel_modeling.ipynb)
- [Shrinkage in hierarchical models](http://doingbayesiandataanalysis.blogspot.com/2012/11/shrinkage-in-multi-level-hierarchical.html)
- [Imperial slide deck on Bayesian hierarchical modeling](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/astrophysics/public/icic/data-analysis-workshop/2018/BHMs.pdf)
- [CMU Hierarchical Models](http://www.stat.cmu.edu/~brian/463-663/week11/Chapter%2009.pdf)
- [Gelman post on BHM](https://statmodeling.stat.columbia.edu/2018/03/24/economist-wrote-asking-make-sense-fit-bayesian-hierarchical-models-instead-frequentist-random-effects/)
- [David Blei on BHM](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/hierarchical-models.pdf)
- [How hierarchical models improve point estimates of model parameters at the individual level](https://www.sciencedirect.com/science/article/pii/S0022249616300025)


# OLD POST
In this post I attempt to answer the question: "what really makes hierarchical models more flexible than non-hierarchical models?". The concept seems relatively straightforward on the surface. Also, if we want a more flexible model, can't we just use flat priors? What's the secret hierarchical sauce?

May need to use a more complex dataset than just simple linear regression. I will need to find a problem where an MCMC struggles to find an appropriate solution with a single prior, but works well with a hierarchical prior.

### Hierarchical Models

In a Bayesian setting, a hierarchical model induces a "hierarchy" into the priors. All this means in practice is that what would have been a fixed parameter in a prior will now come from a distribution.
Consider a prior on some parameter, $\beta \sim N(\mu, \sigma^2)$. In a non-hierarchical setting, we set values for $\mu$ and $\sigma$, however in a hierarchical setting, we place priors on $\mu$ and/or $\sigma$. For example:

$$ \mu \sim f(\alpha) $$
$$ \sigma \sim g(\theta) $$
$$ \beta \sim N(\mu, \sigma^2) $$

So what does this give us? Why would we / should we use a hierarchical model over a non-hierarchical model?

### Why use hierarchical models?

There are a couple different settings where we'll use a hierarchical model, one of the most common is a random effects model (or mixed effects or multilevel models), where the need for hierarchy is to share data between cohorts. Read more [here](https://web.stanford.edu/class/psych252/section/Mixed_models_tutorial.html).

Hierarchical models are most appropriate when we suspect some hierarchy in the data generation process. For example, if we're analyzing grades of students across multiple schools in a country, it would be naive to assume that all classes/schools/regions are the same. This is an example of hierarchical structure in data, students come from schools, which come from regions. 

Three main reasons why we can choose to use a hierarchical model:

1) There is some dependence in the data that isn't necessarily captured by our covariates. For example sampling people from different communities is not independent as individuals in the same community will likely be more similar than individuals across communities. This dependence changes how we consrtuct our likelihood, as the construction of the likelihood assumes data is gathered independently, which is why we can naively multiply the likelihood density. **is this data sharing?**
2) We may believe that the data itself comes from a hierarchy. For example, students come from classes, which are in schools, which are in communities, etc. Modelling students across different classes can be problematic as there is dependence on the environmental context of the class.
3) Finally, we may want to use hierarchical priors to induce flexibility into the model. Although the change is subtle, it changes the posterior enough to yeild different results.

One purpose is methodological; the other is substantive. Methodologically, when units of analysis are drawn from clusters within a population (communities, neighborhoods, city blocks, etc.), they can no longer be considered independent. Individuals who come from the same cluster will be more similar to each other than they will be to individuals from other clusters. Therefore, unobserved variables may induce statistical dependence between observations within clusters that may be uncaptured by covariates within the model, violating a key assumption of maximum likelihood estimation as it is typically conducted when independence of errors is assumed. Recall that a likelihood function, when observations are independent, is simply the product of the density functions for each observation taken over all the observations. However, when independence does not hold, we cannot construct the likelihood as simply. Thus, one reason for constructing hierarchical models is to compensate for the biases—largely in the standard errors—that are introduced when the independence assumption is violated.

In addition to the methodological need for hierarchical models, substantively we may believe that there are differences in how predictors in a regression model influence an outcome of interest across clusters, and we may wish to model these differences. In other words, the influence of predictors may
be context-dependent, a notion that is extremely important and relevant to a social scientific understanding of the world. For example, the emergence of hierarchical modeling in education research occurred because there is a natural nesting of students within classes and classes within schools, schools within communities, and so on, and grades, test performance,
etc. may be dependent on teacher quality, making students in one class different from those in another class. In other words, student performance may be dependent on the teacher—the environmental context of classes.

### Pooling the Right Way 

I won't go into detail about using hierarchical models to pool data, as there are many great blog posts about that. However I'll briefly walk through an example where pooling is necessary. 

Other great posts that talk about this use of hierarchical models are:  

- [Best of both worlds: Hierarchical models](https://twiecki.io/blog/2014/03/17/bayesian-glms-3/)
- [Radon data analysis](https://github.com/fonnesbeck/multilevel_modeling/blob/master/multilevel_modeling.ipynb)
- [Shrinkage in hierarchical models](http://doingbayesiandataanalysis.blogspot.com/2012/11/shrinkage-in-multi-level-hierarchical.html)


### Posterior Overhaul 

Hierarchical models also work because the resulting posterior is changed completely. I'll demonstrate that in this section and walk through an example using a non-hierarchical and a hierarchical model. 

### Open Questions

- **Hierarchical priors make the model more flexible. Why can't we just use a flat prior for added flexibility?** - Two main things here, hierarchical models allow you to pool data from different classes without making naive assumptions about the classes (is this a random effects model?). Second: in general can we just use flatter priors? Or should they be conjugate? Is it because of the resulting posterior?  

- Specifically in the case of Normal-InverseChiSquared, the resulting posterior distribution is t-distributed, which has heavier tails than just a Normal. How will using a Normal-InverseChiSquared hierarchical prior compare to using a non-hierachical prior with fat tails? What if we just use a t-distributed prior? For this maybe we can try the 8 schools model with a prior with fat tails?

- I want to compare these 3 prior specs on an appropriate problem. Appropriate meaning not too simple:
  - Normal prior
  - Flat prior, to try to induce flexibility without hierarchy
  - Hierarchical prior

- In addition to the added flexibility, hierarchical models also help with data pooling. For example in 8 schools.

The $t_{\nu}(\mu, \sigma^2)$ distribution can be represented with a Gaussian with mean $\mu$ and variance term distributed as an Inverse-$\chi^2$ with $\nu$ degrees of freedom -->
