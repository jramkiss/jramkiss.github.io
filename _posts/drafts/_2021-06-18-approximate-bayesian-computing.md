---
layout: post
title: "Introduction to Approximate Bayesian Inference"
date: 2021-06-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: 
---

## What

Approximate Bayesian computing is a family of approximate inference methods targeted at drawing approximate posterior samples when the likelihood, $p(X \mid \theta)$ is computationally intractable but it is easy to sample from the model, $X \sim p(. \mid \theta)$. 

&nbsp;


### Rejection-Sampling 

<!-- An early method for generating approximate random samples from posterior distributions is [rejection-sampling](https://www.genetics.org/content/genetics/145/2/505.full.pdf) by comparing summary statistics from observed data and simulated data. A summary statistic, $S$, is chosen and calculated from the observed data. Then we forward simulate starting from the prior of the parameters of interest to generate a new dataset, which we then calculate the summary statistic, $s'$. Once $s'$ is "close enough" to $S$, we accept the sample. [Beaumont, 2002](https://www.genetics.org/content/genetics/162/4/2025.full.pdf) extends this by perturbing the samples of $\phi_i$ relative to the difference between $s'_i$ and $S$.  -->

##### Rejection-Sampling in an ideal world

An early method for generating samples from approximate posterior distributions is [rejection-sampling](https://www.genetics.org/content/genetics/145/2/505.full.pdf). In an ideal world with a tractable likelihood function, we would first sample a candidate parameter from the prior, $\theta^* \sim p(\theta)$. Then forward-simulate our data, $X^* \sim p(X \mid \theta^*)$, to get a sample from the joint distribution, $(\theta^*, X^*) \sim p(\theta^*, X^*)$. If the simulated data, $X^*$, "matches" the observed data, $X$, we accept $\theta^*$ and repeat the process. After repeating this many times, we end up with approximate samples from the posterior, $p(\theta \mid X)$. 
Now I only have one question... how are we sure this works?? Here's an explanation below: 

$$ p(\theta^*, X^*) = p(X^* \mid \theta^*) p(\theta) \mathbb{I}_X(X^*) $$

Where $\mathbb{I}_X$ is 1 if $X^*$ "matches" $X$, and 0 otherwise. Therefore, averaging $\mathbb{I}_X(X^*)$ for all values of $X^*$ will give the acceptance probability of the our rejection-sampling method, since we only accept when the simulated data matches the observed data. 

Marginalizing $X^*$, we get:

$$ p(\theta^*) = \int p(X^* \mid \theta^*) p(\theta^*) \mathbb{I}_X(X^*)  dX^* $$

$$ p(\theta^*) = p(X \mid \theta^*) p(\theta^*) \propto p(\theta^* \mid X) $$

So the distribution of our resulting $\theta^*$'s is proportional to the posterior distribution we were after in the first place. An intuition for this method is to keep trying to recreate the observed data using different values of $\theta$. For the reconstructions that are really close to the observed data, keep the values of $\theta$, otherwise throw it out. 

&nbsp;


##### Rejection-sampling in practice 

We want to do inference on models with intractable likelihoods, so we need to introduce some more approximations. The posterior conditional on the full dataset, $p(\theta \mid X_{obs})$, is approximated by effectively summarizing $X_{obs}$ with summary statistics, $s_{obs}$. We end up with $p(\theta \mid s_{obs}) \propto p(s_{obs} \mid \theta)p(\theta)$. Since the likelihood is intractable, $p(s_{obs} \mid \theta)$ is also likely intractable and we have to introduce another approximation of $p_{ABC}(\theta \mid s_{obs}) = \int p(\theta, s \mid s_{obs}) ds$.

$$ p(\theta, s \mid s_{obs}) \propto K(\mid \mid s' - s_{obs} \mid \mid) p(s \mid \theta) p(\theta) $$

Where $K$ is a kernel that measures how far away $s'$ is from $s_{obs}$.

From Wikipedia: 
> The outcome of the ABC rejection algorithm is a sample of parameter values approximately distributed according to the desired posterior distribution, and, crucially, obtained without the need to explicitly evaluate the likelihood function.

Below is code for a simulated example using the first rejection scheme described above. This exposes some holes in the rejection-sampling methodology. Firstly, it is very inefficient if $p(\theta)$ is far from $p(\theta \mid X)$, which it usually is. Second, the sampling scheme doesn't take into account previously sampled points. 

```python 
num_tries = 10000
for eps in [0.5]: #, 0.4, 0.3, 0.2, 0.1]:
    # we never call the `likelihood` function
    samples = []
    for _try in range(num_tries):
        # 1) simulate \sigma from prior
        prior_sim = sigma_prior.rvs(size = 1)
        # 2) forward generate data using the value of \sigma just samples
        data_sim = simulator(prior_sim)
        # 3) calculate the sufficient statistic of the simulated data
        sim_ss = sufficient_summary(data_sim)
        # 4) compare the estimated stat and the actual stat. 
        if (distance(sim_ss, data_ss) < eps):
            # accept
            samples.append(prior_sim)
    samples = np.array(samples).T
    fraction_accepted = len(samples) / num_tries
```


### ABC in MCMC




<!-- ## So What

## Now What -->


---

### Questions

- Since ABC methods are targeted at problems with intractable likelihoods, do you need a tractable likelihood for VI? Do you need a tractable likelihood for all other posterior approximations? What about Laplace approximation? 
- In what situations are likelihoods computationally intractable? How could a likelihood be computationally intractable but the model could be easy to sample from? I **really** don't understand this. Isn't sampling from the `model` the same as sampling from the likelihood? 
- Is approximate bayesian computation the same as approximate bayesian inference? It seems like ABC is a sub-field of approximate inference
- What does `explicit likelihood` mean? I see this all over the literature. Does it just mean we don't have a analytic likelihood that we can write down?
- What is an example of a nuisance parameter? Is the hierarchical parameter an example of this? 


### Conditional Density Estimation 

- How are the first 2 papers sure that they produce approximate samples? It seems like there is a lot of intuition here 
- What problem does conditional density estimation solve? And why do we need it?
- What are some ways of estimating conditional densities?



## Annotated Bib

### Normalizing Flows and Conditional Density Estimation

- [Conditional Density Estimation with NN's](https://arxiv.org/pdf/1903.00954.pdf)
- [Gaussian Process Conditional Density Estimation](https://papers.nips.cc/paper/2018/file/6a61d423d02a1c56250dc23ae7ff12f3-Paper.pdf)
- [Annealed Flow Transport Monte Carlo](http://proceedings.mlr.press/v139/arbel21a.html): Combining SCM samplers with Normalizing flows

### Apprroximate Bayesian Computation 
- [Intro to ABC Slides](https://www.maths.lu.se/fileadmin/maths/forskning_research/InferPartObsProcess/abc_slides.pdf): Very comprehensive slides, this is where I got the proof that the rejection-sampling scheme works.
- []
- [Bayesian Computation with Intractable Likelihoods](https://arxiv.org/abs/2004.04620)
- [Jupyter notebook on ABC with code](http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2017/tutorials/aux9_abc.html): Plots and code on ABC from scratch
- [Paper Tutorial on ABC](https://www.semanticscholar.org/paper/A-tutorial-on-approximate-Bayesian-computation-Turner-Zandt/88974ed0ac2f5d9c1c6a0cf04ac7306045540c9c)
- [Approxiimate Bayesian Computation, in a Bio Journal](https://storage.googleapis.com/plos-corpus-prod/10.1371/journal.pcbi.1002803/1/pcbi.1002803.pdf?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=wombat-sa%40plos-prod.iam.gserviceaccount.com%2F20210623%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210623T212921Z&X-Goog-Expires=86400&X-Goog-SignedHeaders=host&X-Goog-Signature=85cc0b8ea6a192df72554cf38a8227930aa0ed069e094e42782cdc53cae796847e8a5024535f90da9da14d26e97ff3882ee521ec196b704db7b0750faa05a0e82a6f6e4fac9109ce21f3eb06982d80a588b1ebc1908e0ab0dfa67f202afe95d2707e100a3f83060a3dd1d7fac1b1dc39c5cb24014bdb463ba3760411855a3277b2901fac0b999136c4d83ee0cbbfa700d0c1f4d986f6b595c508a6d2e844857e872f733f323f799a2c27a733813e669645a650d71e4f9456a4e0079dbd860220604444a67d2abc6f995aa64cc29813c61b25da4e6cd584fadad419b1fa440142ba2f4344bb0ef9714bdf492b2fffc3aae33758e78cc7054c1662cacbcf820ce7)

### Approximate Inference

- [Yingzhen Li Topics in Approximate Inference](http://yingzhenli.net/home/pdf/topics_approx_infer.pdf)
- [Yingzhen Li papers](http://yingzhenli.net/home/en/?page_id=345): Lots of work on approximate inference, with easy to understand "cartoons".
- [Neurips2020 Tutorial on Approximate Inference](http://yingzhenli.net/home/en/?page_id=1341)
- 