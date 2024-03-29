---
layout: post
title: "Introduction to Approximate Bayesian Inference"
date: 2021-06-29 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Intro to ABC
---

## Introduction (What)

Approximate Bayesian computing is a family of approximate inference methods targeted at drawing approximate posterior samples when the likelihood, $p(X \mid \theta)$ is computationally intractable but it is easy to sample from the model, $X \sim p(. \mid \theta)$. These are generally simulation based models, where it is easy to generate $X$ if we have $\theta$, but impossible to write down the likelihood. Some examples of these models are **exmaples here**


The lack of a likelihood function makes Bayesian inference particularly challenging, since the usual way of modelling the posterior is through the product of the likelihood and the priors. To proceed with inference, we have to either directly approximate the posterior, or first approximate the likelihood, then use it to model the posterior. 

Since we can't estimate $L(\theta; X)$, but we can sample from $p(X \mid \theta)$ easily, one direction for inference is as follows: 

1) Propose $\theta_1 \sim p(\theta)$ using the prior on $\theta$ \
2) Simulate data using $p(X_1 \mid \theta_1)$ \
3) If the distance between $X_1$ and $X_{obs}$ is less than $\epsilon$, we accept $\theta_1$, and reject otherwise. We can use summary statistics to define distance here. \
4) Now each accepted $\theta$ is an sample from the approximate posterior $p(\theta \mid X)$. This approximation comes from an approximation to the likelihood, which is dependent on, $\epsilon$. 


This is called a rejection sampling scheme and not surprisingly, it isn't the most efficient or robust. It is sensitive to the choice of summary statistics and threshold, and there is a tradeoff between statistical and computational efficiency. [This paper](https://academic.oup.com/sysbio/article/66/1/e66/2420817) has lots more about it. 

Aside from these drawbacks, if we think about this algorithm a bit, each draw of $\theta$ is independent, and we can probably do more to inform future draws of $\theta$. 

&nbsp;

<!-- ### Rejection-Sampling 


##### Rejection-Sampling in an ideal world

An early method for generating samples from approximate posterior distributions is [rejection-sampling](https://www.genetics.org/content/genetics/145/2/505.full.pdf). In an ideal world with a tractable likelihood function, we would first sample a candidate parameter from the prior, $\theta^{\ast} \sim p(\theta)$. Then forward-simulate our data, $X^{\ast}\sim p(X \mid \theta^{\ast})$, to get a sample from the joint distribution, $(\theta^{\ast}, X^{\ast}) \sim p(\theta^{\ast}, X^{\ast})$. If the simulated data, $X^{\ast}$, "matches" the observed data, $X$, we accept $\theta^{\ast}$ and repeat the process. After repeating this many times, we end up with approximate samples from the posterior, $p(\theta \mid X)$. 
Now I only have one question... how are we sure this works?? Here's an explanation below: 

$$ p(\theta^{\ast}, X^{\ast}) = p(X^{\ast} \mid \theta^{\ast}) p(\theta) I_X(X^{\ast}) $$

Where $I_X$ is 1 if $X^{\ast}$ "matches" $X$, and 0 otherwise. Therefore, averaging $I_X(X^{\ast})$ for all values of $X^{\ast}$ will give the acceptance probability of the our rejection-sampling method, since we only accept when the simulated data matches the observed data. 

Marginalizing $X^{\ast}$, we get:

$$ p(\theta^{\ast}) = \int p(X^{\ast} \mid \theta^{\ast}) p(\theta^{\ast}) I_X(X^{\ast})  dX^{\ast} $$

$$ p(\theta^{\ast}) = p(X \mid \theta^{\ast}) p(\theta^{\ast}) \propto p(\theta^{\ast} \mid X) $$

So the distribution of our resulting $\theta^{\ast}$'s is proportional to the posterior distribution we were after in the first place. An intuition for this method is to keep trying to recreate the observed data using different values of $\theta$. For the reconstructions that are really close to the observed data, keep the values of $\theta$, otherwise throw it out. 

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


### ABC in MCMC -->




<!-- ## So What

## Now What -->


---

### Questions

* Since ABC methods are targeted at problems with intractable likelihoods, do you need a tractable likelihood for VI? Do you need a tractable likelihood for all other posterior approximations? What about Laplace approximation? 
    * We need a likelihood function for the Laplace approximation becasue we approximate the maximum of the posterior, which is the likelihood multiplied by the priro. I think we also need it for VI but it's not clear where it shows up in the ELBO for me right now.


* Do SMC methods require likeilhood functions?
    * Yes, in order to compute the importance weights we need knowledge of $p(y_t \mid x_t, \theta)$
    

* In what situations are likelihoods computationally intractable? How could a likelihood be computationally intractable but the model could be easy to sample from? I **really** don't understand this. Isn't sampling from the `model` the same as sampling from the likelihood? 
    * This happens in simulation models, like SDE's. We have a closed for solution for how the model/latent variables evolve, but no way to right down the likelihood. Not quite sure what the reason for us not being able to write down the likelihood is but I think it has something to do with latent variables and dependence between observations? But then if it has something to do with the dependence between observations, that would imply that we can't write down the likelihood for any reasonably complex time-series model... yea idk.

* Is approximate bayesian computation the same as approximate bayesian inference? It seems like ABC is a sub-field of approximate inference
    * From what I understand, ABC is a subfield of approximate inference specifically dealing with the case where we don't have a tractable likelihood. The other parts of approximate inference deals with approximating the posterior distribution for efficiency reasons, like variational inference or Laplace approximation. 


* What does `explicit likelihood` mean? I see this all over the literature. Does it just mean we don't have a analytic likelihood that we can write down? 
    * I think so.


* What is an example of a nuisance parameter? Is the hierarchical parameter an example of this?
    * I think this is just a latent variable, which we have to integrate out to do inference on the parameters.



## Annotated Bib


### Apprroximate Bayesian Computation 

- [Intro to ABC Slides](https://www.maths.lu.se/fileadmin/maths/forskning_research/InferPartObsProcess/abc_slides.pdf): Very comprehensive slides, this is where I got the proof that the rejection-sampling scheme works.
- [Bayesian Computation with Intractable Likelihoods](https://arxiv.org/abs/2004.04620)
- [Jupyter notebook on ABC with code](http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2017/tutorials/aux9_abc.html): Plots and code on ABC from scratch
- [Paper Tutorial on ABC](https://www.semanticscholar.org/paper/A-tutorial-on-approximate-Bayesian-computation-Turner-Zandt/88974ed0ac2f5d9c1c6a0cf04ac7306045540c9c)
- [Overview of ABC](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1002803&type=printable)
- [Approxiimate Bayesian Computation, in a Bio Journal](https://storage.googleapis.com/plos-corpus-prod/10.1371/journal.pcbi.1002803/1/pcbi.1002803.pdf?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=wombat-sa%40plos-prod.iam.gserviceaccount.com%2F20210623%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210623T212921Z&X-Goog-Expires=86400&X-Goog-SignedHeaders=host&X-Goog-Signature=85cc0b8ea6a192df72554cf38a8227930aa0ed069e094e42782cdc53cae796847e8a5024535f90da9da14d26e97ff3882ee521ec196b704db7b0750faa05a0e82a6f6e4fac9109ce21f3eb06982d80a588b1ebc1908e0ab0dfa67f202afe95d2707e100a3f83060a3dd1d7fac1b1dc39c5cb24014bdb463ba3760411855a3277b2901fac0b999136c4d83ee0cbbfa700d0c1f4d986f6b595c508a6d2e844857e872f733f323f799a2c27a733813e669645a650d71e4f9456a4e0079dbd860220604444a67d2abc6f995aa64cc29813c61b25da4e6cd584fadad419b1fa440142ba2f4344bb0ef9714bdf492b2fffc3aae33758e78cc7054c1662cacbcf820ce7)
- [A Comparison of Likelihood-Free Methods With and Without Summary Statistics](https://arxiv.org/pdf/2103.02407.pdf)
- [https://www.cs.ubc.ca/~arnaud/delmoral_doucet_jasra_smcabc.pdf](https://www.cs.ubc.ca/~arnaud/delmoral_doucet_jasra_smcabc.pdf): Connection between ABC and SMC

### Approximate Inference

- [Yingzhen Li Topics in Approximate Inference](http://yingzhenli.net/home/pdf/topics_approx_infer.pdf)
- [Yingzhen Li papers](http://yingzhenli.net/home/en/?page_id=345): Lots of work on approximate inference, with easy to understand "cartoons".
- [Neurips2020 Tutorial on Approximate Inference](http://yingzhenli.net/home/en/?page_id=1341)
- 

### Normalizing Flows and Conditional Density Estimation

- [Conditional Density Estimation with NN's](https://arxiv.org/pdf/1903.00954.pdf)
- [Gaussian Process Conditional Density Estimation](https://papers.nips.cc/paper/2018/file/6a61d423d02a1c56250dc23ae7ff12f3-Paper.pdf)
- [Annealed Flow Transport Monte Carlo](http://proceedings.mlr.press/v139/arbel21a.html): Combining SCM samplers with Normalizing flows