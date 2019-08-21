---
layout: post
title: "Word Vectors Decomposed - Word2Vec and GloVe"
date: 2019-08-12
category: notebook
comments: true
author: "Jonathan Ramkissoon"
math: true
#markdown:
#  path: 2019-08-12-word2vec_GFM.md
#  ignore_from_front_matter: true
#  absolute_image_path: false
#export_on_save:
#  markdown: true
---


This post will introduce and explain the intuition and math behind [word2vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) and [GloVe](https://nlp.stanford.edu/pubs/glove.pdf). At the end, the reader will be able to explain these algorithms in lehman terms, and have a solid understanding of the math involved. Only very basic math concepts are necessary for the understanding of this post.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

- [Word2Vec](#word2vec)
  - [Overview](#overview)
  - [Deep Dive](#deep-dive)
    - [Skip-Gram Model](#skip-gram-model)
    - [Problems with Naive Softmax](#problems-with-naive-softmax)
    - [Negative Sampling Loss](#negative-sampling-loss)
  - [Word2vec in Python](#word2vec-in-python)
    - [Word Arithmetic](#word-arithmetic)
    - [Visualizing word2vec Embeddings](#visualizing-word2vec-embeddings)
- [GloVe](#glove)
  - [Overview](#overview-1)
  - [Deep Dive](#deep-dive-1)
    - [The Co-Occurrence Matrix](#the-co-occurrence-matrix)
    - [From Word2vec to GloVe](#from-word2vec-to-glove)
- [Fasttext](#fasttext)
- [GloVe VS Word2vec VS Fasttext](#glove-vs-word2vec-vs-fasttext)
- [Appendix](#appendix)
  - [From Word2vec to GloVe - Math](#from-word2vec-to-glove-math)
  - [Code](#code)
- [Resources](#resources)

<!-- /code_chunk_output -->

## Word2Vec

[Word2vec](https://arxiv.org/pdf/1310.4546.pdf) comes in two flavors, the continuous bag-of-words and skip-gram model. These models are similar; CBOW predicts a target word given the context words, and skip-gram predicts context words given the target word. This inversion might seem arbitrary, but it turns out that CBOW smoothes over distributional information by treating an entire context as one observation, which is useful for smaller datasets. Skip-gram on the other hand treats each context-target pair as a new observation, and tends to do better on larger data sets.

### Overview

We focus on the skip-gram model in this post. The model architecture is a 1-layer neural network, whose weights we learn. These weights are then used as the word vectors. The objective is to simultaneously (1) maximize the probability that an observed word appears in the context of it's target word and (2) minimize the probability that a randomly selected word from the vocabulary appears as a context word for the given target word.

* $word2vec('king') - word2vec('man') + word2vec('woman') = word2vec('queen')$  
* $word2vec('doctor') - word2vec('man') + word2vec('woman') = word2vec('nurse')$


### Deep Dive

#### Skip-Gram Model

The skip-gram model is an architecture for learning word embeddings that was first presented in this [paper](https://arxiv.org/pdf/1301.3781.pdf). The idea is that we take a word in the input sequence as the "target" or "center" word, and predict the words around it. 'Around' is determined by a pre-specified window size, $m$.

More formally, we have an input sequence of words, $w_1, w_2,.., w_T$, each of which has a context window around them, $-m \le j \le m$. For each word in the context window, $w_{t+j}$, we calculate the probability that it appears in the context of target word $w_t$. The probability of word $w_{t+j}$ appearing in the context of the given target word $w_t$ can be expressed as $p(w_{t+j} | w_t)$.
The mathematical way of representing this is shown below. Breaking down this equation into its constituent parts and referencing their explanation above will help to understand it.

$$
J(\theta) = -\frac{1}{T} \sum^{T}_{t = 1} \sum_{-m \le j \le m, j \ne 0} log(p(w_{t+j} | w_t; \theta))
$$


The skip-gram model is different from other approaches to word embeddings, such as continuous bag-of-words, which is also presented in the original [skip-gram paper](https://arxiv.org/pdf/1301.3781.pdf). The continuous bag-of-words architecture attempts to predict the target word given its context.

> _PUT A VISUALIZATION OF A SENTENCE, THE TARGET WORD AND WINDOW_

| ![Figure 1: Taken from Stanford's NLP course.](../../assets/word2vec_viz.png) |
|:--:|
| Figure 1: Taken from Stanford's NLP course, shows the skip gram prediction of "banking" with window size 2. |


> SGNS seeks to represent each word w ∈ $V_W$ and each context c ∈ VC as d-dimensional vectors w and ⃗c, such that words that are “similar” to each other will have similar vector representations. It does so by trying to maximize a function of the product w · ⃗c for (w, c) pairs that occur in D, and minimize it for negative examples: (w, cN ) pairs that do not necessarily occur in D. The negative examples are created by stochastically corrupting observed (w, c) pairs from D – hence the name “negative sampling”. For each observation of (w,c), SGNS draws k contexts from the empirical unigram distribution P (c) = #(c).


#### Problems with Naive Softmax

Before we start, recall that an objective or loss function, $J(\theta)$, is a way of determining the goodness of a model. We alter the parameters of this function, $\theta$, to find the best fit for the model. Here we make the ideas about target and context words discussed above more concrete. Note that the parameters of our model are the word embeddings (vectors) we want to find.

The probability, $p(w_{t+j} | w_t; \theta))$, can be expressed using the naive softmax function:

$$
p(w_{t+j} | w_t; \theta) = \frac{e^{u_o^T v_c}}{\sum_{w=1}^W e^{u_w^T v_c}}
$$

Where:

-   $W$ is the size of the vocabulary
-   $c$ and $o$ are indices of the words in the vocabulary at sequence positions $t$ and $j$ respectively
-   $u_o = word2vec(w_{t+j})$
-   $v_c = word2vec(w_t)$

Here, $u_o$ is the "context" vector representation of word $o$ and $v_c$ is the "target" vector representation of word $c$. Having two representations simplifies the calculations, and we can always combine the two representations after.

Although we now have a way of quantifying the probability a word appears in the context of another, the $\sum_{w=1}^W e^{u_w^T v_c}$ term presents difficulty. It requires us to iterate over all words in the vocabulary and do some calculation. The computational complexity of this term is proportional to the size of the vocabulary, which can be massive, more than $10^6$. The authors introduce a clever way of dealing with this called negative sampling.

#### Negative Sampling Loss

Negative sampling overcomes the need to iterate over all words in the vocabulary to compute the softmax by sub-sampling the vocabulary. We sample $k$ words and determine the probability that these words **do not** co-occur with the target word. The idea is to train binary logistic regressions for a true pair versus a couple of noise pairs. According to the paper, this is done because a good model should be able to differentiate between data and noise.

W=To incorporate negative sampling, the skip-gram objective function needs to be altered by replacing $p(w_{t+j} | w_t)$ with:

$$
log(\sigma(u_o^T v_c)) + \sum_{i = 1}^{k}E_{j \sim P(w)} [log(\sigma(-u_j^T v_c))]
$$

Where $\sigma(.)$ is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function).

> Thus the task is to distinguish the target word $w_t$ from draws from the noise distribution $P_n(w)$ using logistic regression, where there are $k$ negative samples for each data sample.

**New Objective Function**

$$
J(\theta) = \frac{-1}{T} \sum_{t=1}^{T}J_t(\theta)
$$

$$
J_t(\theta) = log(\sigma(u_o^T v_c)) + \sum_{i = 1}^{k}E_{j \sim P(w)} [log(\sigma(-u_j^T v_c))]
$$

$$
P(w) = U(w)^{3/4}/Z
$$

This is too much, lets look each component of $J_t(\theta)$ and try to convince ourselves this makes sense.

**_The first part_**, $log(\sigma(u_o^Tv_c))$, can be interpreted as the log "probability" of the target and context words co-occurring. We want our model to find vector representations of $u_o$ and $v_c$ to maximize this probability. The word probability is used loosely here, and is only thrown around because the sigmoid function returns a number between 0 and 1.

**_The second part_**, $\sum_{i = 1}^{k}E_{j \sim P(w)} [log(\sigma(-u_j^T v_c))]$, is where the "sampling" in negative sampling happens. Let's break this up more to make it clearer. It'll come in handy to note that $\sigma(-x) = 1 - \sigma(x)$.
First, we can drop the $E_{j \sim P(w)}$ term, since we already know we will be sampling words from some distribution, $P(w)$

$$
\sum_{i = 1}^{k} log(\sigma(-u_j^T v_c)) = \sum_{i = 1}^{k} log(1 - \sigma(u_j^T v_c))
$$


Now this makes a bit more sense, we're taking the log of 1 minus the probability that the sampled word, $j$, appears in the context of the target word $c$. This is just log of the probability that $j$ does **not** appear in the context of the target word $c$. Since $j$ is a randomly drawn word out of ~$10^6$ words, there's a very small chance it appears in the context of $c$, so this probability should be high.
There is also a summation term here up to $k$ elements. All this is saying is that we're sampling $k$ words from $P(w)$.

Finally, we have to specify a distribution for negative sampling, $P(w) = U(w)^{3/4}/Z$. Here, $U(w)$ is the unigram distribution and is raised to the $\frac{3}{4}$th power to sample rarer words in the vocabulary. $Z$ is just a normalization term.

To summarize, this loss function is trying to maximize the probability that word $o$ appears in the context of word $c$, while minimizing the probability that a randomly selected word from the vocabulary appears in the context of word $c$. We use the gradient of this loss function to update the word vectors, $u_o$ and $v_c$ to get our word embeddings.


**Summary of Word2Vec**

-   Iterate through every word in the whole corpus
-   Predict surrounding words (context words) using word vectors
-   Update the word vectors based on the loss function

### Word2vec in Python

Instead of training our own word2vec model, we'll use a pre-trained model to visualize word embeddings. We'll use Google's News dataset model, which can be downloaded [here](https://code.google.com/archive/p/word2vec/). Fair warning that the model is 1.5Gb, and is trained on a vocabulary of 3 million words, with embedding vectors of length 300.

This model doesn't contain some common words, like "and" or "of", however it does contain others like "the" and "also". [This repo](https://github.com/chrisjmccormick/inspect_word2vec) has a more in-depth analysis of what the model contains and doesn't.

We'll use the `gensim` Python package to load and explore the model. If you dont have it installed, run `pip install gensim` in your command line.  

```python
import gensim

# Download model and save it to current directory, or update the path
model_path = "GoogleNews-vectors-negative300.bin"

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)  

# extract word vectors from the model
wv = model.wv

# remove model from env
del model
```

Now we have vector representations for all words in the vocabulary in `wv` and can query this like a dictionary.

```python
dog_vec = wv["dog"] # "dog" embedding

# Distances from each word in animal_list to the word animal
animal_list = ["dog", "cat", "mouse", "hamster"]
animal_similarity = wv.distances("animal", animal_list)
list(zip(animal_list, animal_similarity))
```
```
[('dog', 0.35619873),
 ('cat', 0.40923107),
 ('mouse', 0.69155),
 ('hamster', 0.5817574)]
```

#### Word Arithmetic

Let's add and subtract some word vectors, then see what the closest word to the resulting vector is. We've all seen the "king" - "man" + "woman" = "queen" example, so I'll present some new ones.

Results generated by the `find_most_similar` function are of the form (word, cosine similarity), where "word" is the closest word to the vector parsed into the function. Cosine similarity values closer to 1 means the vectors (words) are more similar. Its definition can be found in the appendix.
Let's see what the model thinks a `doctor - man + woman` is:

```python
find_most_similar(wv["doctor"] - wv["man"] + wv["woman"],
                  ["man", "doctor", "woman"])
```
```
[('gynecologist', 0.7276507616043091),
 ('nurse', 0.6698512434959412),
 ('physician', 0.6674120426177979)]
```

Interesting, what about if we make a subtle change to `doctor - woman + man`?

```python
find_most_similar(wv["doctor"] - wv["woman"] + wv["man"],
                  ["man", "doctor", "woman"])
```
```
[('physician', 0.6823904514312744),
 ('surgeon', 0.5908077359199524),
 ('dentist', 0.570309042930603)]
```

This is a different results from the original query! Biases in the training data are captured and expressed by the model. I won't go into detail about this here. Instead you should take away that the order of arithmetic for word vectors matters a great deal.


#### Visualizing word2vec Embeddings

To wrap up word2vec, lets look at how the model clusters different words. I've compiled words from different walks of life to see if word2vec was able to unravel their semantic similarities. These words are parsed through word2vec, then the first 2 principal components are plotted. Some expected similarities are seen here, however it should be noted that we lose a lot of information from reducing the dimension from 300 to 2.


```python
# Embedding that makes sense
plot_embeds(["dog", "cat", "hamster", "pet"] +                   # animals
            ["boy", "girl", "man", "woman"] +                    # humans
            ["grown", "adult", "young", "baby"] +                # age
            ["german", "english", "spanish", "french"] +         # languages
            ["mathematics", "physics", "biology", "chemistry"])  # natural sciences
```

![](../../assets/word2vec_pca(1).png)

---

## GloVe  

GloVe (Global Vectors) is another architecture for producing word embeddings. It improves on some key downsides of the skip-gram model, as well as incorporating its advantages. One of these downsides is the loss of corpus statistics due to capturing information one window at a time. GloVe's loss function incorporates word-word occurrence counts to capture global information about context.


One of these downsides of the skip-gram is it tries to capture information from the corpus one window at a time. In doing so, it loses out on key statistical information about the entire corpus, such as word co-occurrence counts. On the other hand, methods that rely solely on co-occurrence counts (eg: SVD on the co-occurrence matrix) fail to capture rich relationships between words. GloVe tries to incorporate the advantages of both the skip-gram model and count-based models.

>  GloVe, for Global Vectors, because the global corpus statistics are captured directly by the model.

### Overview

For each pair of words, GloVe tries to minimize the difference between their dot product and log co-occurrence count.

### Deep Dive

#### The Co-Occurrence Matrix  

We have a matrix, $X$, where each row corresponds to a target word, and each column corresponds to a context word. The entry at $X_{ij}$ is then the number of times word $j$ occurs in the context of word $i$. Context is defined in the same way as the skip-gram model. Summing over all the values in row $i$, will give the number of words that occur in its context, $X_i = \sum_k X_{ik}$. Now we can define $P_{ij}$, the probability of word $j$ occurring in the context of word $i$, as $P(i | j) = \frac{X_{ij}}{X_i}$.

Two main advantages of computing the co-occurrence matrix is that it contains all statistical information about the corpus and only needs to be computed once. We will see how it's used in the next section.

#### From Word2vec to GloVe

The skip-gram model uses negative sampling to bypass the bottleneck of naive softmax loss. GloVe takes a different approach to this, which we'll discuss in this section.

Starting from the naive softmax function, we calculate the probability, $Q_{ij}$, that word $j$ appears in the context of word $i$. Then summing $-log(Q_{ij})$ for each context-target word pair (i.e. each $i$ and $j$) in the corpus can give us the global loss. This is similar to cross-entropy loss.

$$J = - \sum_{i \in corpus} \sum_{j \in context} log (Q_{ij}) $$

Since the words $i$ and $j$ appear $X_{ij}$ times in the corpus, we don't need to iterate over all windows in the corpus, but can iterate over the vocabulary instead. As a refresher, the corpus is the body of text we train the model on, and the vocabulary is all unique words in that text.

$$ J = - \sum_{i = 1}^{W} \sum_{j = 1}^{W} X_{ij} log Q_{ij} $$

Re-arranging some terms, we can come up with this:  

$$ J = - \sum_{i = 1}^{W} X_{i} \sum_{j = 1}^{W} P_{ij}log(Q_{ij}) $$

**What's going on right now?**
- **Where did $P_{ij}$ come from?** - Remember that $P_{ij} = \frac{X_{ij}}{X_i}$ and $X_i = \sum_k X_{ik}$, therefore we can substitute $X_{ij} = P_{ij}X_i$.
- **What's the relationship between $P_{ij}$ and $Q_{ij}$?** - Remember that $P_{ij}$ is the probability that word $j$ appears in the context of word $i$ but $Q_{ij}$ is also the probability that word $j$ appears in the context of word $i$. The difference between the two lies in how they are calculated; $P_{ij}$ is calculated using the data (corpus and vocabulary), so it doesn't change. On the other hand, $Q_{ij}$ is the naive softmax probability, that is calculated using the dot product of word vectors of $i$ and $j$, ($u_j^T v_i$). We have the ability to change $Q_{ij}$ by changing these word vectors.
- **What's the point of $P_{ij}log(Q_{ij})$?** - Now that we've refreshed our memory of $P$ and $Q$, we can see that $P$ is the *true* probability distribution of context and target words, and $Q$ is some made up distribution based on the "goodness" of the word vectors. We really want these two distributions to be close to each other. Observing this, $H = P_{ij}log(Q_{ij})$, when $P$ and $Q$ are close to each other, $H$ is small, and when $P$ and $Q$ are far apart, $H$ gets larger. Our end goal is the minimization of $J$, so the smaller $H$ is is better.

See [Appendix](#appendix) for more

Now we need to find some measure of "closeness" between $P$ and $Q$. We still have $Q_{ij}$ and $P_{ij}$ whose normalization terms we have to iteration over the entire vocabulary to calculate. GloVe overcomes this by dropping the normalization terms completely.


> To begin, cross entropy error is just one among many possible distance measures between probability distributions, and it has the unfortunate property that distributions with long tails are often modeled poorly with too much weight given to the unlikely events. Furthermore, for the measure to be bounded it requires that the model distribution Q be properly normalized. This presents a computational bottleneck. A natural choice would be a least squares objective in which normalization factors in Q and P are discarded

Continuing down the cross-entropy route doesn't work because of the normalization terms for probability distributions. We just discard these normalization terms, and turn the loss into a weighted least squares function.

$$ \hat{J} = \sum_{i = 1}^{W} \sum_{j = 1}^{W} X_{ij} (\hat{P}_{ij} - \hat{Q}_{ij})^2 $$

Where $\hat{Q}_{ij} = e^{u_j^T v_i}$ and $\hat{P}_{ij} = X_{ij}$ are un-normalized probability distributions. The problem with this is that $X_{ij}$ takes on very large values for common words in the vocabulary, like "the", "of", etc. GloVe accounts for this is by taking the log counts. The new objective function then becomes:

$$ \hat{J} = \sum_{w = 1}^{W} \sum_{w = 1}^{W} X_{ij} (u_j^T v_i - log(X_{ij}))^2 $$

We still end up with the normalization factor, $X_{ij}$ which can still suffer from huge values from common words. To deal with this, a weighting function, $f(X)$, is introduced to cap large values.

$$ \hat{J} = \sum_{w = 1}^{W} \sum_{w = 1}^{W} f(X_{ij}) (u_j^T v_i - log(X_{ij}))^2 $$

---

## Fasttext

---

## GloVe VS Word2vec VS Fasttext
- Which is more robust?
- Which is more efficient?
- What does word2vec capture than GloVe doesn't?
-

---

## Appendix

### From Word2vec to GloVe - Math

Naive Softmax function:
$$ Q_{ij} = \frac{e^{u_j^T v_i}}{\sum_{w=1}^W e^{u_w^T v_i}} $$
The bottleneck to the naive softmax function is that the calculation of $\sum_{w=1}^W e^{u_w^T v_i}$ requires iteration over the entire vocabulary.

Summing negative log of $Q_{ij}$ to get the global loss:
$$J = - \sum_{i \in corpus} \sum_{j \in context} log (Q_{ij}) $$

$$ J = - \sum_{i = 1}^{W} X_{i} \sum_{j = 1}^{W} P_{ij}log(Q_{ij}) $$
The term: $\sum_{j = 1}^{W} P_{ij}log(Q_{ij})$ is the cross-entropy of $P_{ij}$ and $Q_{ij}$.

### Code
```Python
# find the 3 most similar words to the vector "vec"
def find_most_similar (vec, words = None) :
    # vec: resulting vector from word Arithmetic
    # words: list of words that comprise vec
    s = wv.similar_by_vector(vec, topn = 10)
    # filter out words like "king" and "man", or else they will be included in the similarity
    if (words != None) :
      word_sim = list(filter(lambda x: (x[0] not in words), s))[:3]
    else :
      return (s[:3])
    return (word_sim)

def plot_embeds(word_list, word_embeddings = None, figsize = (10,10)) :
    # pca on the embedding
    pca = PCA(n_components=2)
    X = pca.fit_transform(wv[word_list])

    ax = plt.figure(figsize=figsize)
    ax.subplots()
    _ = plt.scatter(X[:,0], X[:,1])
    for label, point in list(zip(word_list, X)):
        _ = plt.annotate(label, (point[0], point[1]))
```

---

## Resources
Small [review](https://www.aclweb.org/anthology/Q15-1016) of GloVe and word2vec
[Evaluating](https://www.aclweb.org/anthology/D15-1036) unsupervised word embeddings
Stanford NLP coursenotes on [GloVe](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes02-wordvecs2.pdf)
Stanford NLP coursenotes on [word2vec](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)
[GloVe](https://nlp.stanford.edu/pubs/glove.pdf)
[Stanford NLP coursenotes](http://web.stanford.edu/class/cs224n/index.html#coursework)
[Gensim Models](https://radimrehurek.com/gensim/models/word2vec.html)
[Word2vec in Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
Atom [markdown docs](https://shd101wyy.github.io/markdown-preview-enhanced/#/).
