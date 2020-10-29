---
layout: post
title: "Contrastive Loss"
date: 2020-07-15 12:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: This post explains contrastive loss
---


- What is it used for
- Why use contrastive loss over cross-entropy loss
- Explain the loss function



```Python
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
```


### Readings

- Original paper: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
- https://gombru.github.io/2019/04/03/ranking_loss/
-
