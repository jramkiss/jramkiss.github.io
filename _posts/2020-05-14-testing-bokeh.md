---
layout: post
title: "This is me testing Bokeh"
date: 2020-04-22 12:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: Testing interactive plots with Bokeh
---


## introduction

Text text text

```python
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html

plot = figure()
plot.circle([1,2], [3,4])

html = file_html(plot, CDN, "my plot")

#print(html)
file = open("bokeh_test.html","w")
file.write(html)
file.close()
```

## Bokeh Plot

{% include bokeh_test.html %}
