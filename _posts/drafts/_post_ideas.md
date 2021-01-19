---
layout: post
title: "Post Ideas"
date: 2019-11-01 19:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Bullet points of ideas for future posts.
---


### Can a neural network learn max() and min() functions?
- I want to know if a regular neural network can learn `max()` and `min()` functions. What about a residual neural network?
- First plot the max and min function to see what they look like 

### Can you plot distributions from its moments?
- If you can do this, then we can represent much richer distributinos using variational auto encoders. Instead of learning $(\mu, \sigma)$, you just learn a bunch of moments.
- May be able to use the method of moments to approximate a distribution. Read about this and understand it pleaes
- Reading material: 
    - https://stackoverflow.com/questions/4807398/how-to-generate-distributions-given-mean-sd-skew-and-kurtosis-in-r
    - https://gist.github.com/josef-pkt/6895915
    - https://www.statsmodels.org/stable/gmm.html
    - https://stats.stackexchange.com/questions/375515/can-i-use-moments-of-a-distribution-to-sample-the-distribution



### Hypothesis Testing and Significance Levels
- Basically a post walking through that interview question I got about significance levels and metrics. The math behind what's really going on. And a couple explanations of p-values.
- p-value: False positive rate. How is it a false positive rate

### Gaussian Process Text Classification
- I want to test out out-of-distribution classification on a simple text classification problem. To do this I want to see if the overarching standard of uncertainty calibration is the Gaussian Process.
- Also, it will be cool to learn about Gaussian Processes for classification. I should learn and explain how its different to regression
- Here are some reading resources: [Radford Neal paper](https://www.cs.toronto.edu/~radford/ftp/val6gp.pdf), [GP for ML classification chapter](http://www.gaussianprocess.org/gpml/chapters/RW3.pdf), [GP for classification review](http://www.ideal.ece.utexas.edu/seminar/GP-austin.pdf), [Kaggle thing on GPs](https://www.kaggle.com/residentmario/gaussian-process-regression-and-classification), [more notes](https://www.newton.ac.uk/files/seminar/20080218163017001-151449.pdf)

### Hierarchical Models: Where does the flexibility come from?
- Outline a hierarchical model
- Try to determine where the flexibility comes from. Why is it so much more flexibile than a non-hierarchical model?
- Can use the example of the Normal-Inverse-ChiSquared being the t-distribution, which has fatter tails than just the normal.

### Online Mixture of Gaussians
- Key Idea: Write code to estimate Gaussian mixture models with Dirichlet process priors using $D$ data. Then have a constant stream of data being added, $D_{t+1}...$ and update the GMM online. Plot all of this happening in real time.
- Understand and explain GMM's and online estimation of GMM's
- Figure out how to write an online updating GMM in Python
- figure out how to plot clusters
- Figure out how to constantly update the plot with new points and new clusters. If we can get circles around each cluster that would be ideal.

### Effect of Outliers on Regression Tasks
- Take a dataset, like [this](https://www.kaggle.com/epattaro/brazils-house-of-deputies-reimbursements) and find outliers with simple linear regression. Make a plot and show which points will be considered outliers.
- Then use a robust method for linear regression and show which points whill be considered outliers. Is there a difference?

### Evaluation of Fine-Grained Classification for Document Pages
- How does imagenet pre-training affect document page classification? Does it help?
- Is a document page classification a fine-grained task? If so, does imagenet accuracy correlate? i.e. Do models that perform well on imagenet also perform well on document pages?
- Use this dataset: https://www.cs.cmu.edu/~aharley/rvl-cdip/
-


### Inferring Clusters with Dirichlet Processes
- Implement a dirichlet process clustering model with MNIST
- Explain what Dirichlet processes are
- Can use MNIST with 10 classes, and less than 10 classes and see how it works



### Online Learning with Bayesian Methods
- Read this paper: https://www.ki.tu-berlin.de/fileadmin/fg135/publikationen/opper/Op98b.pdf
- Implement a simple model with online learning and compare the time and accuracy to a model where we re-estimate using the entire dataset everytime we get new data.
- Compare the two models for accuracy, time to predictions and ease of implementations
- Do we have to know about the posterior to do online learning?

### What in the World is a Dirichlet Process
- Basically a small explanation about conjugate priors and the relationship between binonial distributions and beta distributions, then multinomial distributions and dirichlet distributions.
- From there, explain dirichlet processes

### Why Gaussian Process Regression?
- Explain what fitting a GP is
- What is Guassian Process regression and what is the difference between that, regular regression and bayesian regression

### Bayesian Optimization
- Why use Bayesian Optimization
- Can we use bayesian optimization with machine learning? We can start with a ulmfit model and use Bayesian optimization to tune hyperparameters like BPTT, LR, etc.

### Tradeoffs Between Priors and Data
- How do strong priors affect Bayesian models? Is my model going to be subjective if I start with a strong prior that is wrong? Why would I use Bayesian methods if I have no prior beliefs? (all flat priors)
- Do we update the prior during bayesian inference? Or we only update the posterior?
- Walkthrough about the use of priors, when they get used and the effect they have.
    - Are the updated when we update data?
    - If we start with a strong prior that is wrong, how will this affec the model?
    - How long does it take for data to overcome a prior?
- Have a toy example with bayesian linear regression, maybe an example with interpretabble coefficients, like weight VS height. So we can start with a strong prior and not a lot of data, then test out weak priors with lots of data, etc.
