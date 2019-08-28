---
layout: page
title: Posts
description: Notes of JR
permalink: /posts/
---

Exploring machine learning and statistics. 

<ul>
  {% for post in site.categories.notebook %}

    <li>
        <span> {{ post.date | date: "%B, %Y" }} </span> - <a href="{{ post.url }}" title="{{post.title}}"> {{ post.title }} </a>
        <br>
        <blockquote> {{ post.excerpt }} </blockquote>

    </li>

    <br>

  {% endfor %}
</ul>
