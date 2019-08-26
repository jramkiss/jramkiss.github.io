---
layout: page
title: Posts
description: Notes of JR
permalink: /posts/
---

This is where I document my exploration of machine learning and statistics.

<ul>
  {% for post in site.categories.notebook %}

    <li>
        <span> {{ post.date | date: "%B, %Y" }} </span> - <a href="{{ post.url }}" title="{{post.title}}"> {{ post.title }} </a>
        <br>
        <span> &nbsp; {{ post.excerpt }} </span>

    </li>

    <br>

  {% endfor %}
</ul>
