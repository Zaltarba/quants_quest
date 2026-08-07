---
layout: page
title: Search
permalink: /search/
robots: "noindex, follow"
sitemap: false
---

<div class="search-shell">
  <p class="search-lead">Find an article by title, topic, or keyword.</p>
  <label class="visually-hidden" for="search-input">Search articles</label>
  <input type="search" id="search-input" placeholder="Search articles..." autocomplete="off" spellcheck="false">

  <div class="search-suggestions" id="search-suggestions">
    <h2>Browse popular topics</h2>
    <div class="search-topic-links">
      <a href="{{ '/categories/' | relative_url }}#quantitative-finance">Quantitative Finance</a>
      <a href="{{ '/categories/' | relative_url }}#statistics">Statistics</a>
      <a href="{{ '/categories/' | relative_url }}#research">Research</a>
      <a href="{{ '/categories/' | relative_url }}#python">Python</a>
    </div>
    <h2>Recent articles</h2>
    <ul>
      {% assign visible_posts = site.posts | where_exp: "post", "post.hidden != true" %}
      {% for post in visible_posts limit: 4 %}
      <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
      {% endfor %}
    </ul>
  </div>

  <ul id="results-container" aria-live="polite"></ul>
</div>

<script src="{{ '/assets/simple-jekyll-search.min.js' | relative_url }}"></script>
<script>
  (function () {
    var input = document.getElementById('search-input');
    var suggestions = document.getElementById('search-suggestions');

    SimpleJekyllSearch({
      searchInput: input,
      resultsContainer: document.getElementById('results-container'),
      json: '{{ '/search.json' | relative_url }}',
      noResultsText: '<li class="search-result"><div style="padding:18px">No matching articles found.</div></li>',
      searchResultTemplate: '<li class="search-result"><a href="{url}"><img src="{image}" alt="" width="96" height="72" loading="lazy"><div><span class="search-result__meta">{category} &middot; {date}</span><h2>{title}</h2><p>{excerpt}</p></div></a></li>'
    });

    input.addEventListener('input', function () {
      suggestions.hidden = input.value.trim().length > 0;
    });
  }());
</script>
