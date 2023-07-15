---
layout: post
title: "Tricks and Guardrails"
date: 2023-05-11 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Tricks and guardrails when setting up new environments / systems
---

# `rm` Aliasing

Create an alias around the `rm` command so that you don't do anything dumb (like I did today) and delete 4 weeks of work in 1 go. 

```bash
alias rm="rm -i"
```

# 