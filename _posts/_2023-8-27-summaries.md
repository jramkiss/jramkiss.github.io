- Eager execution VS graph execution: PyTorch and Tensorflow have these 2 different execution types. Eager execution excecutes commands immediately and returns values, which is good for researchers. Graph mode builds a computational graph that can be run later

- [Optimizing PyTorch model performance](https://pytorch.org/blog/optimizing-production-pytorch-performance-with-graph-transformations/): `torch.FX` provides methods for "operator fusion", which fuses operators to reduce latency at inference time. Can be used to speed up models in production

