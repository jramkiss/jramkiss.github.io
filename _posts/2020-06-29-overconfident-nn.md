---
layout: post
title: "Wrong and Strong Neural Networks"
date: 2020-06-29 12:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: This post is on the overconfidence problem in neural networks
---

I've been struggling with a seemingly simple problem. The task is to build an image classifier to determine if an arbitrary image is sheet music or not. Just like you, on the surface I thought this would be an easy and borderline mundane task - how could this possibly not work??

As a reminder of how "easy" this problem is, here's are example images of sheet music and not sheet music.

<p align="center">
  <img src="/assets/maybe-sheet-music.jpg" height="350">
  <img src="/assets/yes-sheet-music.jpg" height="350">
</p>

&nbsp;

To my surprise, in the evaluation stage of the project, I noticed incredibly high confidence scores
