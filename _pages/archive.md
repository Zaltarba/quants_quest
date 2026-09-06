---
eyebrow: "The collection"
subtitle: "Every public post, gathered in one place."
page_style: archive
description: "Browse the Quants Quest article archive: essays on quantitative finance, statistics, deep learning, and Python, organized by year."
layout: page
permalink: /archive/
title: Journal archive
---

{% assign visible_posts = site.posts | where_exp: "post", "post.hidden != true" %}
{% assign posts_by_year = visible_posts | group_by_exp: "post", "post.date | date: '%Y'" %}
<div class="archive-list">
  {% for year in posts_by_year %}
  <section class="archive-year">
    <h2>{{ year.name }}</h2>
    <ul>
      {% for post in year.items %}
      <li>
        <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %e" }}</time>
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </li>
      {% endfor %}
    </ul>
  </section>
  {% endfor %}
</div>
