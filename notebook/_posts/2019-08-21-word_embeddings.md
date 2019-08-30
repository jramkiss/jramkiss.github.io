---
layout: post
title: "Word Vectors Decomposed - Word2Vec and GloVe"
date: 2019-08-21
category: notebook
comments: true
author: "Jonathan Ramkissoon"
mathjax: true
excerpt: This post is about word embeddings. It explains word2vec, GloVe and fasttext in detail and shows how to use pre-trained models for each algorithm in Python using `gensim`.
---


Accurately representing words as vectors is a challenging, but necessary task in deep learning. Consider the following sentences:

- The garden is pretty
- The garden is pretty ugly

How can a vector represent "pretty", when it means different things in different contexts?

This post will explain 3 breakthrough algorithms for learning word vectors (embeddings), and provide code examples for getting started with pre-trained models in Python. We'll start with [word2vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), which is the oldest of the 3, then explore ways of dealing with its shortcomings in [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) and [fasttext](https://arxiv.org/pdf/1607.04606.pdf).

Why use pre-trained models? - Pre-trained models are great because we don't need a ton of resources to use powerful algorithms. Some of the models used here were trained by Google and Facebook on hundreds of millions of words. Pretty much impossible on a laptop CPU.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=3 orderedList=false} -->
<!-- code_chunk_output -->

- [Word2Vec](#word2vec)
  - [Overview](#overview)
  - [Deep Dive](#deep-dive)
- [GloVe](#glove)
  - [Overview](#overview-1)
  - [Deep Dive](#deep-dive-1)
- [Fasttext](#fasttext)
  - [Overview](#overview-2)
  - [Deep Dive](#deep-dive-2)
- [Open Questions](#open-questions)
- [Word Embeddings in Python](#word-embeddings-in-python)
  - [Word2vec](#word2vec-1)
  - [GloVe](#glove-1)
  - [Fasttext](#fasttext-1)
- [Conclusion](#conclusion)
- [Appendix](#appendix)
  - [From Word2vec to GloVe - Math](#from-word2vec-to-glove-math)
  - [Code](#code)
- [Resources](#resources)

<!-- /code_chunk_output -->


## Word2Vec

### Overview

[Word2vec](https://arxiv.org/pdf/1310.4546.pdf) really refers to one of two models for learning word vectors; the continuous bag-of-words (CBOW) or the skip-gram model. They are very similar, CBOW accepts context words as input and predicts the target word whereas the skip-gram accepts a target word as input and predicts a context word.

> This inversion of predicting context / target words between CBOW and skip-gram might seem arbitrary, but it turns out that CBOW smoothes over distributional information by treating an entire context as one observation (useful for smaller datasets). Skip-gram on the other hand treats each context word - target word pair as a new observation, and tends to do better on larger data sets.

Although we will primarily focus on the skip-gram, both models are single layer neural networks that accept one-hot encoded vectors as input. We learn the weights of the hidden layer, and each row of this weight matrix is a word vector. **The model learns by simultaneously: (1) maximizing the probability that an observed word appears in the context of a target word and (2) minimizing the probability that a randomly selected word from the vocabulary doesn't appear in the context of the target word.**

If you're still unsure about neural network weights and the weight matrix, I recommend reading [this chapter](http://neuralnetworksanddeeplearning.com/chap1.html).


### Deep Dive

The main idea behind the skip-gram model is we take a word in an input sequence as the target word, and predict its context words. The context of a word is the $m$ words surrounding it. In figure 1, the window size is 2, the target word ("into") is in red and its context words ("problems", "turning", "banking", "crises") are in blue.

<br/>

![Figure1](/assets/word2vec_viz.png)
Figure 1: The skip gram prediction of target word "into" with window size 2. Taken from Stanford's NLP course, shows
 <br/>

Before we start, let's formalize some notation. We have an input sequence of words, $w_1, w_2,.., w_T$, each of which has a context window, $-m \le j \le m$. We'll call this input sequence the *corpus*, and all its unique words the *vocabulary*. Each word in the vocabulary will have 2 vector representations, $u_o$ for context and $v_c$ for target.
In figure 1, $u_{turning}$ is the vector representation of "turning" as a context word, and $v_{banking}$ is the vector representation of "banking" as a target word.

We want to calculate the probability that each word in the window, $w_{t+j}$, appears in the context of the target word $w_t$. We'll refer to this probability as $p(w_{t+j} \lvert w_t; \theta)$.  
This may seem weird, but the probability is based on the vector representations of each word. When we encounter a word in the context of another, we alter their vector representations to be "closer". So the more we see words in each other's context, the close their vectors are. Now back to word vectors, the function $J(\theta)$ below describes this; $\theta$ is a placeholder representing all the vector representations.

> To see the intuition, we can forget about vectors: "turning" is more likely to be in the context of "into" than "crises" is (I can think of a million sentences with "turning into", but not that many with "crises into" or "into crises"). So the vectors for "turning" and "into" should be to be closer than "crises" and "into".

$$
J(\theta) = -\frac{1}{T} \sum^{T}_{t = 1} \sum_{-m \le j \le m, j \ne 0} log(p(w_{t+j} \lvert w_t; \theta))
$$

The only problem here is we have no idea how to find $p(w_{t+j} \lvert w_t; \theta)$. We'll start with using the softmax function. This essentially calculates how similar a context word, $u_o$, is to target word $v_c$, relative to all other context words in the vocabulary. The measure of similarity between two words is measured by the dot product $u_o^T v_c$, and a larger dot product means more similar words.

$$
p(w_{t+j} \lvert w_t; \theta) = \frac{e^{u_o^T v_c}}{\sum_{w=1}^W e^{u_w^T v_c}}
$$

Where:

-   $W$ is the size of the vocabulary
-   $c$ and $o$ are indices of the words in the vocabulary at sequence positions $t$ and $j$ respectively
-   $u_o = word2vec(w_{t+j})$
-   $v_c = word2vec(w_t)$

Although we now have a way of quantifying the probability a word appears in the context of another, the $\sum_{w=1}^W e^{u_w^T v_c}$ term requires us to iterate over all words in the vocabulary. To deal with this, we must approximate the softmax probability. One way of doing this is called negative sampling.

<br/>

#### Negative Sampling Loss

Negative sampling overcomes the need to iterate over all words in the vocabulary to compute the softmax by sub-sampling the vocabulary. We sample $k$ words and determine the probability that these words **do not** co-occur with the target word. The intuition behind this is that a good model should be able to differentiate between data and noise.

To incorporate negative sampling, the objective function needs to be altered by replacing $p(w_{t+j} \lvert w_t)$ with:

$$
log(\sigma(u_o^T v_c)) + \sum_{i = 1}^{k}E_{j \sim P(w)} [log(\sigma(-u_j^T v_c))]
$$

Where $\sigma(.)$ is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function).

> Thus the task is to distinguish the target word $w_t$ from draws from the noise distribution $P_n(w)$ using logistic regression, where there are $k$ negative samples for each data sample.

<br/>

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

Let's look each component of $J_t(\theta)$ and try to convince ourselves this makes sense.

**The first part,** $log(\sigma(u_o^Tv_c))$, can be interpreted as the log probability of the target and context words co-occurring. We want the model to find $u_o$ and $v_c$ to maximize this probability.

**The second part,** $\sum_{i = 1}^{k}E_{j \sim P(w)} [log(\sigma(-u_j^T v_c))]$, is where the negative sampling happens. Let's break this up more to make it clearer. It'll come in handy to note that $\sigma(-x) = 1 - \sigma(x)$.

We can first drop the $E_{j \sim P(w)}$ term, since we already know we will be sampling words from some distribution, $P(w)$. We end up with:

$$
\sum_{i = 1}^{k} log(\sigma(-u_j^T v_c)) = \sum_{i = 1}^{k} log(1 - \sigma(u_j^T v_c))
$$

Now this is easier to read, we're taking the log of 1 minus the probability that the sampled word, $j$, appears in the context of the target word $c$. This is just log of the *probability that $j$ does **not** appear in the context of the target word* $c$. Since $j$ is randomly drawn out of ~$10^6$ words, there's a very small chance it appears in the context of $c$, so this probability should be high. We do this for each of the $k$ sampled words.

Finally, we have to specify a distribution for negative sampling, $P(w) = U(w)^{3/4}/Z$. Here, $U(w)$ is the count of each word in the corpus (unigram distribution) and is raised to the $\frac{3}{4}$th power to sample rarer words in the vocabulary. $Z$ is just a normalization term to turn $P(w)$ into a probability distribution.

To summarize, this loss function is trying to maximize the probability that word $o$ appears in the context of word $c$, while minimizing the probability that a randomly selected word from the vocabulary does not appear in the context of word $c$. We use the gradient of this loss function to iteratively update the word vectors, $u_o$ and $v_c$ and eventually get our word embeddings.


**Summary of Word2Vec**

-   Iterate through every word in the whole corpus
-   Predict surrounding words (context words) using word vectors
-   Update the word vectors based on the loss function

<br/>

---

## GloVe

### Overview

GloVe (Global Vectors) is another architecture for learning word embeddings that improves on the skip-gram model by incorporating corpus statistics. Since the skip-gram model looks at each window independently, it loses corpus statistics. Instead, GloVe uses word co-occurrence counts to capture global information about the corpus. **GloVe learns word embeddings by minimizing the difference between word vector dot products and their log co-occurrence counts.**

> On the other hand, methods that rely solely on co-occurrence counts (eg: SVD on the co-occurrence matrix) fail to capture rich relationships between words. GloVe tries to incorporate the advantages of both the skip-gram model and count-based models.

### Deep Dive

#### The Co-Occurrence Matrix

The co-occurrence matrix, $X$, is generated from the corpus and vocabulary. The entry at $X_{ij}$ is then the number of times word $j$ occurs in the context of word $i$. Context is defined in the same way as the skip-gram model. Summing over all the values in row $i$, will give the number of words that occur in its context, $X_i = \sum_k X_{ik}$. Then the probability of word $j$ occurring in the context of word $i$ is $P(i \lvert j) = \frac{X_{ij}}{X_i}$.

Below is the co-occurrence matrix for the corpus containing:

- "I like deep learning."
- "I like NLP."
- "I enjoy flying."

![](/assets/cooccurrence_matrix.png)

#### From Softmax to GloVe

We can find global loss using the softmax function, $Q_{ij}$, by summing over all target-context word pairs.

$$J = - \sum_{i \in corpus} \sum_{j \in context} log (Q_{ij}) $$

Since words $i$ and $j$ appear $X_{ij}$ times in the corpus, we don't need to iterate over all windows in the corpus, but can iterate over the vocabulary and multiply by the co-occurrence count.

$$ J = - \sum_{i = 1}^{W} \sum_{j = 1}^{W} X_{ij} log Q_{ij} $$

Re-arranging some terms, we can come up with this:

$$ J = - \sum_{i = 1}^{W} X_{i} \sum_{j = 1}^{W} P_{ij}log(Q_{ij}) $$

**What's going on right now?**

- **Where did $P_{ij}$ come from?** - Remember that $P_{ij} = \frac{X_{ij}}{X_i}$ and $X_i = \sum_k X_{ik}$, therefore we can substitute $X_{ij} = P_{ij}X_i$.
- **What's the relationship between $P_{ij}$ and $Q_{ij}$?** - $P_{ij}$ is the probability that word $j$ appears in the context of word $i$, but $Q_{ij}$ is also the probability that word $j$ appears in the context of word $i$. The difference between the two lies in how they are calculated. $P_{ij}$ is calculated using the co-occurrence matrix and doesn't change. $Q_{ij}$ is the naive softmax probability, that is calculated using the dot product of word vectors $u_j$ and $v_i$. We have the ability to change $Q_{ij}$ by changing these vectors.
- **What's the point of $P_{ij}log(Q_{ij})$?** - Now that we've refreshed our memory of $P$ and $Q$, we can see that $P$ is the *true* probability distribution of context and target words, and $Q$ is some made up distribution based on the "goodness" of the word vectors. We really want these two distributions to be close to each other. Observing $H = P_{ij}log(Q_{ij})$, when $P$ and $Q$ are close to each other, $H$ is small, and when $P$ and $Q$ are far apart, $H$ is larger. Our end goal is the minimization of $J$, so the smaller $H$ is is better. This is term is the cross-entropy between distributions $P$ and $Q$.

> Cross entropy error is just one among many possible distance measures between probability distributions, and it has the unfortunate property that distributions with long tails are often modeled poorly with too much weight given to the unlikely events.

The problem here is that cross-entropy requires normalized versions of $Q_{ij}$ and $P_{ij}$ which we have to iterate over the entire vocabulary to calculate. This is the reason for using Negative Sampling in the skip-gram model. GloVe's approach to this is dropping the normalization terms completely, so we end up with $\hat{P}$ and $\hat{Q}$. The cross-entropy function is now useless, so we change $H = P_{ij}log(Q_{ij})$ to a squared error function, $(\hat{P}_{ij} - \hat{Q}_{ij})^2$.

$$
\hat{J} = \sum_{i = 1}^{W} X_{i} \sum_{j = 1}^{W} (\hat{P}_{ij} - \hat{Q}_{ij})^2
$$

Now we have squared error, weighted by the number of co-occurrences of words $i$ and $j$. There's one last problem with this, which is that some co-occurrence counts can be massive. This will affect both the weights, $X_i$, and $\hat{P_{ij}} = X_{ij}$. To deal with this explosion in the squared term, we take $log(hat{P})$ and $log(hat{Q})$) and to deal with the explosion of weights, we introduce a function, $f$ that caps the co-occurrence count weight. We'll apply $f$ to each target-context pair, $X_{ij}$ as opposed to only $X_i$. The new loss function becomes:

$$
\hat{J} = \sum_{w = 1}^{W} \sum_{w = 1}^{W} f(X_{ij}) (u_j^T v_i - log(X_{ij}))^2
$$

This is the loss function that the GloVe model minimizes.

See [Appendix](#appendix) for more

---

## Fasttext

### Overview

[Fasttext](https://github.com/facebookresearch/fastText) is a powerful library for learning word embeddings that was introduced by Facebook in 2016. The package offers text classification as well. Its roots come from the [word2vec](#deep-dive) models.

Word2vec trains a unique vector for each word, ignoring important word sub-structure (morphological structure) and making out-of-vocabulary prediction nearly impossible. Fasttext attempts to solve this by treating each word as a sum of its subwords. These subwords can be defined in any way, however the simplest form is a character n-gram. Then the vector for each word is simply the sum of each of its n-grams.

**Fasttext learns word embeddings the same way as word2vec, but treats each word as a sum of n-grams instead of one unit.**

> This is especially significant for morphologically rich languages (German, Turkish) in which a single word can have a large number of morphological forms, each of which might occur rarely, thus making it hard to train good word embeddings.

### Deep Dive

Before starting, we'll take a step back to quantifying how similar two word vectors are. Both GloVe and word2vec do this using dot products, however we can think of the similarity more generally as an arbitrary function, $s(u_j, v_c)$.

Fasttext uses a different similarity measure, where each word is represented as a sum of smaller words of length n called n-grams. To help the model learn prefixes and suffixes, we append "<" to the front and ">" to the back of each word. Then for n=3, the n-grams of "where" are:

    <where> = [<wh, whe, her, ere, re>]

We have no way of determining the difference between the subword "her" and the full-word "her" (there definitely should be a difference). Appending the special characters around each word helps with this, as the tri-gram "her" is now different from the sequence "\<her>".

More formally, suppose we have all n-grams in the vocabulary, $G$, each represented by a vector, $\boldsymbol{z}_g$. We can refer to all the n-grams of some word, $w$, by $G_w$. Then $w$ can be represented as a sum of all n-grams. The new similarity function becomes:


$$ s(w, c) = \sum_{g \in G_w} \boldsymbol{z}_g^T v_c$$


---

## Open Questions

- **Fasttext** doesn't account for positional arrangement of sub-words. Are there any examples of words with the same subwords in different context that contribute to the word differently?

- We can introduce this scoring function to **GloVe** as well. Has this been done?

- In **word2vec**, do we do any downsampling of frequent words before training? Not referring to negative sampling

- Do we do downsampling of frequent words in **fasttext**? [This post](https://towardsdatascience.com/fasttext-under-the-hood-11efc57b2b3) talked about it, but I didn't see anything in the paper, might be in the source code.


---

## Word Embeddings in Python

Now let's explore word embeddings using pre-trained models in the `gensim` Python package. If you don't have it installed, run `pip install gensim` in your command line. Gensim offers pre-trained models from their `gensim.downloader` method and each model used here embeds words in a 300-dimensional space. A full list of the models available can be found [here](https://github.com/RaRe-Technologies/gensim-data), or by running `python -m gensim.downloader --info` in your command line.


I provided some functions in the [Appendix](#appendix) for plotting and finding most similar words that are built on top of `Gensim` method. Here're the imports we'll need:


```python
import gensim
import gensim.downloader as api

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# clearer images if you're using a Jupyter notebook
%config InlineBackend.figure_format='retina'
```

### Word2vec

For word2vec, we'll use Google's News dataset model, trained on news articles with a vocabulary of 3 million words and 300 dimensional embedding vectors. [This repo](https://github.com/chrisjmccormick/inspect_word2vec) has an in-depth analysis of the words in the model.

```Python
word2vec_model_path = "word2vec-google-news-300"
print(api.info(word2vec_model_path))

word2vec_model = api.load(word2vec_model_path)
w2v = word2vec_model.wv

# remove from env
del word2vec_model
```

### GloVe

The glove-wiki-gigaword-300 model used here is trained on 6B tokens from W ikipedia 2014 and the Gigaword dataset, other pre-trained GloVe models can be downloaded [from Stanford](https://nlp.stanford.edu/projects/glove/) or [from Gensim](https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100).

```python
glove_model_path = "glove-wiki-gigaword-300"
print(api.info(glove_model_path))

glove_model = api.load(glove_model_path)
glove = glove_model.wv

del glove_model
```

### Fasttext

Fasttext provides pre-trained models on for multiple languages, which can be used in different ways (through the command line, downloading the model, through `gensim`, etc.). We'll use the English model provided by `Gensim` which is trained on wikipedia 2017 and news data, but you can go through [their Github](https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md) to see more.


#### Word Comparison

Now we have vector representations for all words in the vocabulary in `wv` and can compare the different models. We'll add and subtract some word vectors, then see what the closest word to the resulting vector is. Papers and blog posts have exhausted the "king" - "man" + "woman" = "queen" example, so I'll present some new ones.

Results generated by the `find_most_similar` function are of the form (word, cosine similarity), where the each word is the closest to the one parsed into the function. Cosine similarity values closer to 1 means the vectors (words) are more similar. The function definition can be found in the [Appendix](#appendix).

Start with: `doctor - man + woman`

```python
print ("word2vec: Doctor - Man + Woman")
find_most_similar(w2v["doctor"] - w2v["man"] + w2v["woman"], w2v,
                  ["man", "doctor", "woman"])

print ("GloVe: Doctor - Man + Woman")
find_most_similar(glove["doctor"] - glove["man"] + glove["woman"], glove,
                  ["man", "doctor", "woman"])

print ("fasttext: Doctor - Man + Woman")
find_most_similar(fasttext["doctor"] - fasttext["man"] + fasttext["woman"], fasttext,
                  ["man", "doctor", "woman"])
```

```
 word2vec: Doctor - Man + Woman
 [('gynecologist', 0.7276507616043091),
  ('nurse', 0.6698512434959412),
  ('physician', 0.6674120426177979)]  

 GloVe: Doctor - Man + Woman
 [('physician', 0.6203880906105042),
  ('nurse', 0.6161285638809204),
  ('doctors', 0.6017279624938965)]  

 fasttext: Doctor - Man + Woman
 [('gynecologist', 0.6874127388000488),
  ('nurse-midwife', 0.6773605346679688),
  ('physician', 0.6561880111694336)]
```

Interesting, what about if we make a subtle change to `doctor - woman + man`?

```python
print ("word2vec: Doctor - Woman + Man")
find_most_similar(w2v["doctor"] - w2v["woman"] + w2v["man"], w2v,
                  ["man", "doctors", "doctor", "woman"])

print ("GloVe: Doctor - Woman + Man")
find_most_similar(glove["doctor"] - glove["woman"] + glove["man"], glove,
                  ["man", "doctors", "doctor", "woman", "dr."])

print ("fasttext: Doctor - Woman + Man")
find_most_similar(fasttext["doctor"] - fasttext["woman"] + fasttext["man"], fasttext,
                  ["man", "doctors", "doctor", "woman", "dr."])
```

```
word2vec: Doctor - Woman + Man
[('physician', 0.6823904514312744),
 ('surgeon', 0.5908077359199524),
 ('dentist', 0.570309042930603)]  

GloVe: Doctor - Woman + Man  
[('physician', 0.5128607153892517),
 ('he', 0.4661550223827362),
 ('brother', 0.46356332302093506)]  

fasttext: Doctor - Woman + Man
[('physician', 0.6969557404518127),
 ('docter', 0.6826808452606201),
 ('non-doctor', 0.6698156595230103)]
```

This is a different results from the original results. Biases in the training data are expressed by the model. Also interestingly, there are some misspelled words in fasttext. This is because of the difference in learning methods.


#### Visualizing Embeddings

For the sake of completeness, I plotted words from different walks of life to see the algorithms were able to unravel their semantic similarities/differences. The first 2 principal components of each word vector are plotted. Some expected similarities are seen here, however we lose a lot of information from reducing the dimension from 300 to 2.


```python
plot_embeds(["dog", "cat", "hamster", "pet"] +                   # animals
            ["boy", "girl", "man", "woman"] +                    # humans
            ["grown", "adult", "young", "baby"] +                # age
            ["german", "english", "spanish", "french"] +         # languages
            ["mathematics", "physics", "biology", "chemistry"],  # natural sciences
            w2v,
            title = "word2vec Embedding")
# run this again, but changing w2v to glove and fasttext
```

![](/assets/word2vec_embedding.png)
![](/assets/glove_embedding.png)
![](/assets/fasttext_embedding.png)



## Conclusion

- Which is more robust?
- Which is more efficient?
- What does word2vec capture than GloVe doesn't?
- How does word2vec calculate word embbeddings?
- How does GloVe calculate word embeddings?
- How does fasttetxt calculate word embeddings?

---

## Appendix

**Objective Function:**
Recall that an objective or loss function, $J(\theta)$, is a way of determining the goodness of a model. We alter the parameters of this function, $\theta$, to find the best fit for the model.

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
# some word2vec examples
dog_vec = wv["dog"] # "dog" embedding

# Distances from each word in animal_list to the word animal
animal_list = ["dog", "cat", "mouse", "hamster"]
animal_similarity = wv.distances("animal", animal_list)
list(zip(animal_list, animal_similarity))


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
- Small [review](https://www.aclweb.org/anthology/Q15-1016) of GloVe and word2vec
- [Evaluating](https://www.aclweb.org/anthology/D15-1036) unsupervised word embeddings
- Stanford NLP coursenotes on [GloVe](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes02-wordvecs2.pdf)
- Stanford NLP coursenotes on [word2vec](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)
- [GloVe](https://nlp.stanford.edu/pubs/glove.pdf)
- [Stanford NLP coursenotes](http://web.stanford.edu/class/cs224n/index.html#coursework)
- [Gensim Models](https://radimrehurek.com/gensim/models/word2vec.html)
- [Word2vec in Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
- [GloVe Blog post](https://machinelearningmastery.com/develop-word-embeddings-python-gensim/)
- [Fasttext paper](https://arxiv.org/pdf/1607.04606.pdf)

- Atom [markdown docs](https://shd101wyy.github.io/markdown-preview-enhanced/#/).
- Jekyll [cheatsheet](https://devhints.io/jekyll).
