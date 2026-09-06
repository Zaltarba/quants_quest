---
eyebrow: "Explore the journal"
subtitle: "Follow an idea through finance, statistics, research, and code."
page_style: topics
description: "Explore research notes by topic: quantitative finance, statistics, time series, deep learning, algorithmic trading, and Python."
layout: page
permalink: /categories/
title: Topics
wide: true
---

{% assign sorted_categories = site.categories | sort %}
<div class="topics-browser" data-topics-browser>

  <nav class="topics-browser__nav" aria-label="Choose a topic">
    {% for category in sorted_categories %}
      {% assign category_name = category | first %}
      {% assign category_posts = site.categories[category_name] | where_exp: "post", "post.hidden != true" %}
      {% if category_posts.size > 0 %}
      <a href="#{{ category_name | slugify }}" data-topic-link="{{ category_name | slugify }}">{{ category_name }}</a>
      {% endif %}
    {% endfor %}
  </nav>

  <a class="topics-browser__all" href="{{ '/categories/' | relative_url }}">&larr; All topics</a>

  <div class="topics-browser__collections">
    {% for category in sorted_categories %}
      {% assign category_name = category | first %}
      {% assign category_posts = site.categories[category_name] | where_exp: "post", "post.hidden != true" %}
      {% if category_posts.size > 0 %}
      <section class="topic-group topic-collection" id="{{ category_name | slugify }}" data-topic-group>
        <div class="topic-collection__header">
          <div>
            <h2>{{ category_name }}</h2>
            <p class="topic-group__count">{{ category_posts.size }} {% if category_posts.size == 1 %}article{% else %}articles{% endif %}</p>
          </div>
          <a class="topic-collection__open" href="#{{ category_name | slugify }}">View articles {% include arrow.html %}</a>
        </div>

        <div class="topic-entries article-list">
          {% for post in category_posts %}
            {% include article_row.html post=post %}
          {% endfor %}
        </div>
      </section>
      {% endif %}
    {% endfor %}
  </div>
</div>

<script>
  (function () {
    var browser = document.querySelector('[data-topics-browser]');
    if (!browser) return;

    var groups = Array.prototype.slice.call(browser.querySelectorAll('[data-topic-group]'));
    var links = Array.prototype.slice.call(browser.querySelectorAll('[data-topic-link]'));

    function showTopic() {
      var requested = '';
      try { requested = decodeURIComponent(window.location.hash.slice(1)).toLowerCase(); } catch (error) { /* Show all topics for a malformed URL. */ }
      var selected = groups.find(function (group) { return group.id.toLowerCase() === requested; });

      browser.classList.toggle('is-topic-index', !selected);
      browser.classList.toggle('is-topic-selected', Boolean(selected));
      groups.forEach(function (group) { group.hidden = Boolean(selected) && group !== selected; });
      links.forEach(function (link) {
        var active = Boolean(selected) && link.dataset.topicLink.toLowerCase() === requested;
        link.classList.toggle('is-active', active);
        if (active) link.setAttribute('aria-current', 'true');
        else link.removeAttribute('aria-current');
      });

      if (selected) {
        window.requestAnimationFrame(function () {
          selected.scrollIntoView({ block: 'start' });
        });
      }
    }

    window.addEventListener('hashchange', showTopic);
    showTopic();
  }());
</script>
