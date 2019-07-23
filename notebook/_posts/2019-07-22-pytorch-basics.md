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
PyTorch, and PyTorch users who want a refresher of the basics. It will introduce PyTorch basics, then implement linear regression in both Numpy and PyTorch to compare the two. We'll finish by implementing a simple feed foward neural network to predict the famous MNIST dataset.

If you've had some exposure to PyTorch before, you may want to skip the [PyTorch Basics](#pytorch-basics) section.

- [PyTorch Basics](#pytorch-basics)
- [Linear Regression in PyTorch and Numpy](#Linear-Regression-in-PyTorch-and-Numpy)

# PyTorch Basics

[PyTorch](http://pytorch.org/) is a framework for building and training deep learning models. It integrates easily in the python ecosystem because of how much it mimmincs Numpy. While the primary datastructure in Numpy is the array, the primary datastructure in PyTorch is a `tensor`. In real life a tensor is just a generalization of a matrix, and you can have fun with the formal definition [here](http://mathworld.wolfram.com/Tensor.html).

For more clarity, a vector is just a 1-dimensional tensor, a matrix is a 2-dimensional tensor (greyscale images) and a 3-dimensional array is a 3-dimensional tensor (RGB images).

If you're a 100% newby (as we all were at some point), you'll want to start by first downloading PyTorch. Run `pip install torch` in your command line to do so.

### Defining Tensors

Now that we have PyTorch installed, we can start getting our feet wet by defining
some tensors.

```python
import torch

# what does a tensor look like..
torch.tensor([1,2,3])
```

```
tensor([1, 2, 3])
```

What about higher dimesnional tensors?

```python
torch.zeros((2, 3)) + 2 # same as torch.zeros(2, 3) + torch.tensor(2)
```

```
tensor([[2., 2., 2.],
        [2., 2., 2.]])
```

```python
# and 3-dimensions, filled with random numbers
torch.randn((3, 3, 3)) # same as torch.randn(3, 3, 3)
```

```
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

### PyTorch Math

Lets define more tensors so we can do math on them. We'll multiply 2 matricies, then add a scalar and finally sum the resulting tensor.

```python
a = torch.randn((3))
a.shape # check the size to be sure: torch.Size([3])

b = torch.randn((1, 3))
b.size() # same as b.shape

c = torch.randn((1, 1))

torch.sum(torch.matmul(a, b.view((3, 1))) + c)
```

What'd we do there?

- `torch.randn((N))`: Create a tensor of dimension N and fill it with random numbers
- `torch.matmul(_, _)`: Matrix multiplication. PyTorch provides many functions for matrix multiplication. `torch.mm(_, _)` can also be used. You can read more about this function [here](https://pytorch.org/docs/stable/torch.html#torch.matmul).
- `view()`: In order to multiply `a` and `b` in the order we want, we must match their dimensions appropriately. PyTorch provides some other ways of doing this: `b.reshape()` and `b.resize_()`, however `b.view()` is usually the most reliable. In Numpy this can be done with `.reshape()`.
- `torch.sum()`: Sums the elements of a tensor and returns a tensor of shape 1


# Linear Regression in PyTorch and Numpy

To illustrate PyTorch without neural networks, let's implement linear regression in both Numpy and PyTorch.


```Python
print("this is a syntax highlighting test")
```
