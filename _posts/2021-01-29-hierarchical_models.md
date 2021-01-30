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

In this post I build a Bayesian hierarchical model to classify [Amazon products from Kaggle](https://www.kaggle.com/kashnitsky/hierarchical-text-classification) into a taxonomy using their titles. The taxonomy is hierarchical in nature and we can benefit from incorporating this structure into a model. First we'll look at the taxonomy and class membership, then talk about simple approaches to the classification problem. Finally, we'll build the Bayesian model and write it up in Numpyro.

&nbsp;

## Hierarchical Class Structure 

Below is a diagram of a subset of parent classes on the left, and child classes on the right. For this problem we have $6$ parent classes and $64$ child classes.

<div class='figure' align="center">
    <img src="/assets/amazon_taxonomy.png" width="50%" height="30%">
    <div class='caption' width="40%" height="40%">
        <p> Taxonomy structure for 2 parent classes (on the left) and 5 of their children classes (on the right). This is a small subset of parent and children classes. </p>
    </div>
</div>

&nbsp;

A simple way of assigning categories to our data would be to flatten the hierarchy and only consider the child classes. This would turn our problem into a $64$ class classification problem which can be solved with logistic regression / SVM / many other things. The drawback of this approach is that we make the assumption that each class is independednt, which we know is incorrect. 

We want to teach the model that "action toy figures" and "baby toddler toys" come from the same parent class. Intuitively, this can help when we don't have a lot of training data for a particular child class, and if we come across an item at inference time that we don't have a subclass for. Thankfully, one way to deal with these problems is by using hierarchical modelling. 


## Data and Preprocessing

Before diving into the model, here is the data we're working with:

<div class='figure' align="center">
    <img src="/assets/Amazon-taxonomy-data.png" width="90%" height="90%">
</div>

&nbsp;

We'll use TF-IDF scores of item titles to classify them into the taxonomy. I used `Gensim` for calculating TF-IDF, but `sklearn` is fine. I found that `Gensim` is more memory efficient when working with _much_ larger datasets. 


```python
%%time
# find tf-idf scores for training set
dct = Dictionary(X_train.map(lambda x: x.split(' ')))
dct.filter_extremes(no_below=5, no_above=0.7, keep_n = 2 ** 10)
dct.compactify()

train_corpus = [dct.doc2bow(doc.split(' ')) for doc in X_train]  # BoW format
tfidf_model = TfIdfTransformer()

train_tfidf = tfidf_model.fit_transform(train_corpus)
train_tfidf = corpus2dense(train_tfidf, num_terms = len(dct)) # can also use: corpus2csc
print(train_tfidf.shape)

test_corpus = [dct.doc2bow(doc.split(' ')) for doc in X_test] 
test_tfidf = corpus2dense(tfidf_model.transform(test_corpus), num_terms = len(dct))
print(test_tfidf.shape)
```
&nbsp;

Finally we encode the labels for parent and children classes in 2 different ways: `labelEncoder` and `labelBinarizer`. The former maps each class into an integer, which we'll use to fit a couple `sklearn` models to compare against our Bayesian model. The latter one-hot encodes.  

```python
le = preprocessing.LabelEncoder()
parent_target = le.fit_transform(parent_train)
children_target = le.fit_transform(children_train)

parent_target_test = le.fit_transform(parent_test)
children_target_test = le.fit_transform(children_test)

lb = LabelBinarizer()
parent_binr = lb.fit_transform(parent_train)
children_binr = lb.fit_transform(children_train)
```

&nbsp;

## Bayesian Hierarchical Modeling

The class structure can be explicitly represented by our priors in a hierarchical model, so let's do that. First assume that the underlying model is a logistic regression with target, $y$, being the children classes. Therefore $\beta \in R^{p \times c}$, where $p$ is the number of features in the regression and $c$ is the number of children classes, so $64$ in this problem.

$$ Z = X \beta + \epsilon $$

$$ y = \text{softmax}(Z) $$ 

Column $i$ of $\beta$ corresponds to the coefficients for class $i$. We know that class $i$ is the child of parent class $p_i$, and that $i$ has "brothers" which also come from parent class $p_i$. We want each child of parent $p_i$ to have the same prior, which we can represent below:


$$
\begin{equation*}
  \beta_p \sim N(\beta_{\mu_p}, \sigma_{\mu_p}^2) \\[10pt]
  \beta_c = \beta_p \times \alpha \\[10pt]
  \beta \sim N(\beta_c, \sigma_c^2) \\[10pt]
  \epsilon \sim N(0, 1) \\[10pt]
\end{equation*}
$$


Here, $\beta_{\mu_p}$ is the prior mean for each parent class, which we set by hand and $\beta_c$ is the prior mean for each child class. We transform $\beta_p$ into $\beta_c$ by multiplying by another matrix, $\alpha \in R^{p \times c}$. This $\alpha$ is what links the children classes together, and to their parents. 


## Inference in Numpyro

[Numpyro](http://num.pyro.ai/en/latest/getting_started.html) is another probabilistic programming language built on Pyro and [JAX](https://jax.readthedocs.io/en/latest/). It's supposed to blazing fast thanks to speed ups provided by JAX, so we'll try it out here. 


```python
def hierarchical_model (X, Y=None):
    dim_X = X.shape[1]
    # hierarchical prior: beta_0 ~ N(0, 1) 
    beta_0 = numpyro.sample("beta_0", dist.Normal(jnp.zeros((dim_X, np.array([num_parent_classes])[0])), 
                                                  jnp.ones((dim_X, np.array([num_parent_classes])[0]))))
    # construct prior for $\beta$ by multiplying with \alpha 
    jnp_prior_mean = jnp.matmul(beta_0, alpha)
    # now we can sample beta        
    beta = numpyro.sample("beta", dist.Normal(jnp_prior_mean, jnp.ones(jnp_prior_mean.shape)))
    err = numpyro.sample("err", dist.Normal(0., 0.5))
    resp = jnp.matmul(X, beta) + err
    probs = softmax(resp) 
    numpyro.sample("Y", dist.Multinomial(probs = probs), obs = Y)


# functions for inference and prediction: 
# helper function for HMC inference
def run_inference(model, rng_key, X, Y, num_warmup = 10, num_samples = 100, num_chains = 2):
    start = time.time()
    kernel = NUTS(model)
    print("Starting MCMC: ")
    mcmc = MCMC(kernel, num_warmup = num_warmup, 
                num_samples = num_samples, 
                num_chains = num_chains)
    mcmc.run(rng_key, X, Y)
    print('\nMCMC elapsed time:', time.time() - start)
    return mcmc

# helper function for prediction
def predict(model, rng_key, samples, X):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None)
    return model_trace['Y']['value']
```
&nbsp;

Numpyro provides a `reparam` function, to change hierarchical model specifications from centered to non-centered parameterizations. We'll use this to help with inference.

```python
# constructing \alpha
alpha = np.zeros((len(parent_class_list), len(children_class_list)))
for i, c in enumerate(children_class_list) :
    alpha[:, i] = parent_class_list == class_tree[c]

num_parent_classes = len(parent_class_list)
X = jnp.asarray(train_tfidf.T, dtype = "float32")
Y = jnp.asarray(children_binr, dtype = "float32") # binarized labels

_num_chains = 4
_num_samples = 200
numpyro.set_host_device_count(_num_chains)
rng_key, rng_key_predict = random.split(random.PRNGKey(0))

reparam_model = reparam(hierarchical_model, config={'beta': LocScaleReparam(0)})

non_centered_mcmc = run_inference(model = reparam_model, 
                           rng_key = rng_key, 
                           X = X, Y = Y, 
                           num_warmup = 50, 
                           num_samples = _num_samples, 
                           num_chains = _num_chains)
nc_samples = non_centered_mcmc.get_samples()
print("MCMC complete")
```

```
Train set accuracy, child categories:  0.7351668726823238
Test set accuracy, child categories:  0.6277708071936429
```


## Comparison to Flat Classification 

Of course, we need to compare this hierarchical model against simpler formulations. One of the simplest approaches to this problem is to flatten the hierarchy, which we do by disregarding the parent classes. We'll compare a Ridge classifier, SVM classifier and logistic regression:

```python
# rige classifier
ridge_clf = RidgeClassifier().fit(train_tfidf.T, children_target)
print("Ridge classifier train set accuracy, children classes: ", ridge_clf.score(train_tfidf.T, children_target))
print("Ridge classifier test set accuracy, children classes: ", ridge_clf.score(test_tfidf.T, children_target_test))

print('')

# linear SVM
sgd_clf = SGDClassifier(loss = "hinge", # linear SVM, "log" for logistic regression
                        max_iter=1000, 
                        n_jobs = -1, 
                        random_state = 42,
                        tol=1e-3).fit(train_tfidf.T, children_target)
print("SVM classifier train set accuracy, children classes: ", sgd_clf.score(train_tfidf.T, children_target))
print("SVM classifier test set accuracy, children classes: ", sgd_clf.score(test_tfidf.T, children_target_test))

print('')

# logistic regression
log_reg = SGDClassifier(loss = "log", # linear SVM, "log" for logistic regression
                        max_iter=1000, 
                        n_jobs = -1, 
                        random_state = 42,
                        tol=1e-3).fit(train_tfidf.T, children_target)
# log_reg.predict_proba(test_tfidf.T)
print("Logistic Regression train set accuracy, children classes: ", log_reg.score(train_tfidf.T, children_target))
print("Logistic Regression test set accuracy, children classes: ", log_reg.score(test_tfidf.T, children_target_test))
```
&nbsp;

```
Ridge classifier train set accuracy, children classes:  0.7184796044499382
Ridge classifier test set accuracy, children classes:  0.6273525721455459

SVM classifier train set accuracy, children classes:  0.7481458590852905
SVM classifier test set accuracy, children classes:  0.6288163948138854

Logistic Regression train set accuracy, children classes:  0.7081788215904409
Logistic Regression test set accuracy, children classes:  0.6286072772898369
```

On a macro scale, these results are comparable to the hierarchical model. However, the data is very imbalanced, so accuracy won't suffice. To determine how much of a difference the hierarchical model makes, I'm interested in a coulpe things: 

- Performance of parent categories: I expect the hierarchical model to perform well on the parent-level categories, since we implicitly model these in our regression. In particular I'm interested in the parent-level preformance for classes that have few datapoints. 
- Out-of-distribution examples: Because our model is Bayesian, it should also be able to detect out-of-distribution samples better than the frequentist models. We can test this by applying a random text dataset to the models.

&nbsp;

--- 


## Bloopers: Parent Posterior as Children Prior

I also experimented with another formulation where I first fit a logistic regression to predict the parent classes, and obtained the posterior mean for $\beta_p$. I then used $\beta_p$ as the prior mean for another regression, where I predict the children classes. Th formulation turned out to not work as well as the traditional hierarchical model, and I suspect it is because when using the posterior mean of $\beta_p$, the posterior variance was disregarded. This no longer made the model hierarchical, but simply just changed the prior mean for $\beta_c$.

The code I used to run this is below:

```python
# first fit parent regression
def parent_model(X, Y=None):
    if Y == None : 
        dim_Y = 6
    else :
        dim_Y = Y.shape[1]
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros((X.shape[1], dim_Y)), jnp.ones((X.shape[1], dim_Y))*2)) 
    err = numpyro.sample("err", dist.Normal(0., 1.))
    resp = jnp.matmul(X, beta) + err
    probs = softmax(resp) # jax softmax, not scipy
    numpyro.sample("Y", dist.Multinomial(probs = probs), obs = Y)


X = jnp.asarray(train_tfidf.T, dtype = "float32")
Y = jnp.asarray(parent_binr, dtype = "float32") # binarized labels

_num_chains = 4
_num_samples = 200
numpyro.set_host_device_count(_num_chains)
rng_key, rng_key_predict = random.split(random.PRNGKey(0))
mcmc = run_inference(model = parent_model, 
                     rng_key = rng_key, 
                     X = X, Y = Y,
                     num_warmup = 50, 
                     num_samples = _num_samples, 
                     num_chains = _num_chains)
parent_samples = mcmc.get_samples()
print("MCMC complete")

# predict Y_test at inputs X_test
vmap_args = (parent_samples, random.split(rng_key_predict, _num_chains * _num_samples))
predictions = vmap(lambda samples, rng_key: predict(parent_model, rng_key, samples, X))(*vmap_args)

# compute mean prediction and confidence interval around median
mean_prediction = jnp.mean(predictions, axis=0)
percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)

class_predictions = pd.DataFrame(mean_prediction).apply(np.argmax, axis = 1)
print("Training set accuracy: ", np.mean(class_predictions == parent_target))
```
&nbsp;


Now construct the prior mean for the child regression by using the posterior mean of the parent regression.

```python
# use the posterior of the parent regression as the prior mean for the child regression
beta_posterior = parent_samples["beta"]
posterior_mean = np.apply_along_axis(np.mean, 0, beta_posterior)

posterior_mean_df = pd.DataFrame(posterior_mean, columns = parent_class_list)

prior_mean = pd.DataFrame()
for c in children_class_list :
    prior_mean[c] = posterior_mean_df[class_tree[c]]
    
prior_mean.head()
```
&nbsp;

And re-run a similar model, but this time using the children labels as the target:

```python
# model for the child classes
def child_model(X, Y=None):
    beta = numpyro.sample("beta", dist.Normal(jnp_prior_mean, jnp.ones(jnp_prior_mean.shape)*2))
    err = numpyro.sample("err", dist.Normal(0., 0.5))
    resp = jnp.matmul(X, beta) + err
    probs = softmax(resp) # jax softmax, not scipy
    numpyro.sample("Y", dist.Multinomial(probs = probs), obs = Y)

X = jnp.asarray(train_tfidf.T, dtype = "float32")
Y = jnp.asarray(children_binr, dtype = "float32") # binarized labels
jnp_prior_mean = jnp.asarray(prior_mean, dtype = "float32")

_num_chains = 4
_num_samples = 200
numpyro.set_host_device_count(_num_chains)
rng_key, rng_key_predict = random.split(random.PRNGKey(0))

child_model_reparam = reparam(child_model, config={'beta': LocScaleReparam(0)})

child_mcmc = run_inference(model = child_model_reparam, 
                           rng_key = rng_key, 
                           X = X, Y = Y,
                           num_warmup = 50, 
                           num_samples = _num_samples, 
                           num_chains = _num_chains)
child_samples = child_mcmc.get_samples()
print("MCMC complete")


# predict Y_test at inputs X_test
vmap_args = (child_samples, random.split(rng_key_predict, _num_chains * _num_samples))
children_predictions = vmap(lambda samples, rng_key: predict(child_model, rng_key, samples, X))(*vmap_args)

# compute mean prediction and confidence interval around median
mean_children_prediction = jnp.mean(children_predictions, axis=0)
percentiles = np.percentile(children_predictions, [5.0, 95.0], axis=0)

child_class_predictions = pd.DataFrame(mean_children_prediction).apply(np.argmax, axis = 1)
print("Train set accuracy, child categories: ", np.mean(child_class_predictions == children_target))
```

&nbsp;

The test set accuracy of the child model using this formulation was around 44%, significantly worse than both the frequentist models and non-centered hierarchical model.

---

## Helpful Resources

- [Finally! Bayesian Hierarchical Modelling at Scale - Florian Wilhelm](https://florianwilhelm.info/2020/10/bayesian_hierarchical_modelling_at_scale/)
- [Massively parallel MCMC with JAX](https://rlouf.github.io/post/jax-random-walk-metropolis/)



<!-- 

# Outstanding Questions
- How do we actually share data in the hierarchical model? What happen if we dont have \alpha in this model?




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
