---
layout: post
title: "Bayesian Hierarchical Classification in Numpyro"
date: 2021-01-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: In this post I use Numpyro to build a Bayesian model to classify Amazon products into a hierarchical taxonomy.
---

## Introduction

In this post we'll build a Bayesian model to classify Amazon products into a taxonomy using their titles. This taxonomy, like many others, is hierarchical in nature and we can benefit from incorporating this structure into the model. First we'll look at the taxonomy and class membership, then talk about simple approaches to the classification problem. Finally, we'll build the Bayesian model and write it up in Numpyro.

&nbsp;

## Hierarchical Class Structure 

Before we do anything, we should understand the problem better. Below is a plot of a subset of classes to see their structure. It's pretty self explanatory, but each item is a member of a child class, and each child class is a member of a parent class. In this particular problem we have $6$ parent classes and $64$ child classes.


<div class='figure' align="center">
    <img src="/assets/amazon_taxonomy.png" width="80%" height="70%">
    <div class='caption' width="70%" height="70%">
        <p> Taxonomy structure for 2 parent classes (on the left) and 5 of their children classes (on the right). This is a small subset of parent and children classes. </p>
    </div>
</div>
&nbsp;


Before we get to the Bayesian model, we should explore simpler ways for this classification to be done. One approach would be to flatten the taxonomy and only consider the children classes. This turns the probblem into a $64$ class classification problem which can be solved with logistic regression / SVM / anything, really. The main drawback of this approach is that we make the assumption that each class is indepentent, and not only do we know this is incorrect, we have the dependencies. We want to teach the model that "action toy figures" and "baby toddler toys" come from the same parent class. Intuitively, this can help when we don't have a lot of training data for a particular child class, and if we come across an item after training that we don't have a subclass for. For example if we come across an item that should be in the class "adult lego toys", but we don't have that child class defined, we should be able to determine that this item came frm the "toy games" parent class. Thankfully, the way to deal with these problems is hierarchical modelling. 


## Bayesian Hierarchical Modeling

The class structure can be explicitly represented by our priors in a Bayesian hierarchical model, so let's do that. 

First assume that the underlying model is a logistic regression with target, $y$, being the children classes. Therefore $\beta \in R^{p \times c}$, where $p$ is the number of features in the regression and $c$ is the number of children classes, so $64$ in this problem.

$$ Z = X \beta + \epsilon $$

$$ \text{softmax}(Z) = y $$ 

Column $i$ of $\beta$ corresponds to the coefficients for class $i$. We know that class $i$ is the child of parent class $p_i$, and that $i$ has "brothers" which also come from parent class $p_i$. Then each sibling of parent $p_i$ should have the same prior. We can represent that below:

$$ \beta_0 \sim Normal(\, \sigma_0^2) $$

$$ \beta \sim Normal(\beta_0, \sigma_0^2) $$


$$
\begin{equation*}
  \begin{split}
    Z = X \beta + \epsilon \\[10pt]
    y = \text{softmax(Z)}
  \end{split}
  \text{, } \qquad \qquad
  \begin{split}
    \beta_p \sim Normal(\beta_p, \sigma_p^2) \\[10pt]
    \beta_c = \beta_p \times \alpha \\[10pt]
    \beta \sim Normal(\beta_c, \sigma_c^2) \\[10pt]
    \epsilon \sim N(0, 1) \\[10pt]
  \end{split}
  \\[15pt]
\end{equation*}
$$



## Helpful Resources

- [Finally! Bayesian Hierarchical Modelling at Scale - Florian Wilhelm](https://florianwilhelm.info/2020/10/bayesian_hierarchical_modelling_at_scale/)

--- 


<!-- 

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
