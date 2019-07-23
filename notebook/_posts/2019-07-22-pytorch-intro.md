---
layout: post
title: "First Steps With PyTorch - Pweave"
date: 2019-07-22
category: notebook
comments: true
author: "Jonathan Ramkissoon"
math: true
---

This notebook is targeted at Python programmers who want to get started in
PyTorch, and PyTorch users who want a refresher of the basics. It will introduce
PyTorch basics, then implement linear regression in both Numpy and PyTorch to
compare the two. We'll finish by implementing a simple feed foward neural network
to predict the famous MNIST dataset.

If you've had some exposure to PyTorch before, you may want to skip the
[PyTorch Basics](#pytorch-basics) section.

- [PyTorch Basics](#pytorch-basics)
- [Linear Regression in PyTorch and Numpy](#Linear-Regression-in-PyTorch-and-Numpy)

# PyTorch Basics

[PyTorch](http://pytorch.org/) is a framework for building and training deep
learning models. It integrates easily in the python ecosystem because of how
much it mimmincs Numpy. While the primary datastructure in Numpy is the array,
the primary datastructure in PyTorch is a `tensor`. In real life a tensor is
just a generalization of a matrix, and you can have fun with the formal definition
[here](http://mathworld.wolfram.com/Tensor.html).

For more clarity, a vector is just a 1-dimensional tensor, a matrix is a
2-dimensional tensor (greyscale images) and a 3-dimensional array is a 3-dimensional
tensor (RGB images).

If you're a 100% newby (as we all were at some point), you'll want to start by
first downloading PyTorch. Run `pip install torch` in your command line.

Now that we have PyTorch installed, we can start getting our feet wet.



```python
import torch

# what does a tensor look like?
x1 = torch.tensor([1,2,3])
print(x1)
```

```
tensor([1, 2, 3])
```


```python
# tensor the same shape as x1, but with random numbers:
x2 = torch.randn_like(x1)
```

```
---------------------------------------------------------------------------RuntimeError
Traceback (most recent call last)<ipython-input-1-13cffdd61c7d> in
<module>()
----> 1 x2 = torch.randn_like(x1)
RuntimeError: _th_normal_ not supported on CPUType for Long
```


```python
print(x2)
```

```
---------------------------------------------------------------------------NameError
Traceback (most recent call last)<ipython-input-1-223cd8a8d439> in
<module>()
----> 1 print(x2)
NameError: name 'x2' is not defined
```



What about higher dimesnional tensors? Here's how we can create 2 and 3 dimensional
tensors and fill them with random numbers.


```python
x3 = torch.randn(3, 3)
print(x3)
```

```
tensor([[-0.8489, -1.3318,  0.2898],
        [ 0.0040, -1.3582, -0.7617],
        [ 0.1654, -1.6722,  0.8319]])
```


```python
# and 3-dimensions?
x4 = torch.randn(3, 3, 3)
print(x4)
```

```
tensor([[[ 1.1768e-01,  1.0128e-01, -1.1197e+00],
         [ 9.4527e-04,  8.3153e-01,  1.1791e+00],
         [ 2.6623e+00,  8.0585e-01,  4.3461e-01]],

        [[-3.4099e-01,  1.8556e+00,  6.9014e-01],
         [-7.6856e-01,  8.1887e-01,  9.1728e-01],
         [-1.6746e+00,  9.0893e-01, -2.2167e-01]],

        [[-4.5973e-01,  5.6127e-01,  3.2835e-01],
         [ 7.5159e-01, -1.7231e+00,  3.6136e-01],
         [ 4.4655e-01, -6.2232e-01, -5.5383e-01]]])
```




# Linear Regression in PyTorch and Numpy

To illustrate PyTorch without neural networks, let's implement linear regression
in both Numpy and PyTorch.
