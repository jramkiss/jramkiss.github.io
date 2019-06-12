---
layout: page
title: Swimming
description: Thoughts about a previous life as a swimmer
permalink: /swimming/
---

Thoughts on a previous life as a swimmer.

<ul>
  {% for post in site.categories.life %}
    <li>
        <span>{{ post.date | date_to_string }}</span> » <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
        <meta name="description" content="{{ post.summary | escape }}">
        <meta name="keywords" content="{{ post.tags | join: ', ' | escape }}"/>
    </li>
  {% endfor %}
</ul>
