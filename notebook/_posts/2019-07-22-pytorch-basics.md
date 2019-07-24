---
layout: post
title: "First Steps With PyTorch"
date: 2019-07-22
category: notebook
comments: true
author: "Jonathan Ramkissoon"
math: true
---

This notebook is targeted at Python programmers who want to get started in
PyTorch, and PyTorch users who want a refresher of the basics. We will introduce the basics of PyTorch, then compare implementations of linear regression in both Numpy and PyTorch. Finally, we'll fit a simple feed forward neural network to the famous MNIST dataset.

If you've had some exposure to PyTorch before, you may want to start at the [Linear Regression in PyTorch and Numpy](#linear-regression-in-pytorch-and-numpy) section.

- [PyTorch Basics](#pytorch-basics)
- [Linear Regression in PyTorch and Numpy](#linear-regression-in-pytorch-and-numpy)

<br>
---


## PyTorch Basics

[PyTorch](http://pytorch.org/) is a framework for building and training deep learning models. It was designed to integrate seamlessly into the Python ecosystem, and work well alongside standard Python libraries, like Numpy. While the primary data structure in Numpy is the array, the primary data structure in PyTorch is a `tensor`. In real life, a tensor is just a generalization of a matrix (which is a generalization of a vector), and you can have fun with the formal definition [here](http://mathworld.wolfram.com/Tensor.html).

For clarity, a vector is just a 1-dimensional tensor, a matrix is a 2-dimensional tensor (greyscale images for example) and a 3-dimensional array is a 3-dimensional tensor (RGB images for example).

If you're a 100% newby (as we all were at some point), you'll want to start by first downloading PyTorch. Run `pip install torch` in your command line to do so.

#### Defining Tensors

Now that we have PyTorch installed, we can start getting our feet wet by defining
some tensors.

```python
import torch

# what does a tensor look like..
torch.tensor([1,2,3])
```

```Text
tensor([1, 2, 3])
```

What about higher dimesnional tensors?

```python
torch.zeros((2, 3)) + 2 # same as torch.zeros(2, 3) + torch.tensor(2)
```

```Text
tensor([[2., 2., 2.],
        [2., 2., 2.]])
```

```python
# and 3-dimensions, filled with random numbers
torch.randn((3, 3, 3)) # same as torch.randn(3, 3, 3)
```

```Text
tensor([[[-0.1910, -0.8819, -0.2358],
         [ 0.2093, -0.4744,  0.3972],
         [-0.5294, -1.4190, -1.4044]],

        [[ 0.8187,  1.2440,  0.2051],
         [-0.4238,  1.2906,  0.6045],
         [ 1.7416,  0.6723,  1.1501]],

        [[-0.3537,  0.0446,  0.2172],
         [-0.0568, -1.1947,  0.1400],
         [-0.5285, -0.6376,  0.1383]]])
```

#### PyTorch Math

Let's define more tensors so we can start doing math with them. We'll multiply 2 matricies, then add a scalar and finally sum the resulting tensor.

```python
a = torch.randn((3))
a.shape # check the size to be sure: torch.Size([3])

b = torch.randn((1, 3))
b.size() # same as b.shape

c = torch.randn((1, 1))

torch.sum(torch.matmul(a, b.view((3, 1))) + c)
```

What'd we do there?

- [`torch.randn()`](https://pytorch.org/docs/stable/torch.html#torch.randn): Create a tensor of dimension N and fill it with random numbers
- [`torch.matmul()`](https://pytorch.org/docs/stable/torch.html#torch.matmul): Matrix multiplication. PyTorch provides many functions for matrix multiplication. [`torch.mm()`](https://pytorch.org/docs/stable/torch.html#torch.matmul) can also be used.
- [`torch.Tensor.view()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view): In order to multiply two matricies in the order we want (in this case `a` and `b`), we must reshape one tensor so that their dimensions match appropriately. PyTorch also has `torch.Tensor.reshape()` and `torch.Tensor.resize_()`, however `torch.Tensor.view()` is usually the most reliable. In Numpy this can be done with `.reshape()`.
- [`torch.sum()`](https://pytorch.org/docs/stable/torch.html#torch.sum): Sums the elements of a tensor and returns a tensor of shape 1

<br>
---


## Linear Regression in PyTorch and Numpy

To illustrate PyTorch without neural networks, let's implement linear regression from scratch in both Numpy and PyTorch. We'll see how much faster using PyTorch tensors are to Numpy arrays.

#### Linear Regression Refresher

We have data $$X$$, continuous response $$y$$ and want to find $$\beta$$ that minimizes the "distance" between $$f(X)$$ and $$y$$.

$$
f(X) = \beta_0 + \sum_{j=1}X_j \beta_j
$$

Turns out there's a nice formula to find the best $$\beta$$ for this task:

$$
\beta = (X^{T}X)^{-1}X^{T}y
$$

#### Linear Regression in PyTorch

```python
import Torch

torch.manual_seed(10)

torch_X = torch.randn((10, 5))
torch_X[:, 0] = 1 # set first columnn of X to be 1 for the bias term

torch_y = torch.randn(10)

start = time.time()
torch_beta_1 = torch.inverse(torch.matmul(torch.t(torch_X), torch_X))
torch_beta_2 = torch.matmul(torch.t(torch_X), torch_y)
torch_beta = torch.matmul(torch_beta_1, torch_beta_2)
end = time.time()

print("Elapsed time for PyTorch: " + str(end - start))
```

```text
Elapsed time for PyTorch: 0.002402067184448242
```

#### Linear Regression in Numpy

```python
import numpy as np
import torch

np.random.seed(10)

# convert the tensor defined above to numpy to have the same data
np_X = torch_X.numpy()
np_y = torch_y.numpy()

print("this is a syntax highlighting test")
```
