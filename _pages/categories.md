---
layout: page
permalink: /categories/
title: Topics
wide: true
---

{% assign sorted_categories = site.categories | sort %}
<div class="topics-browser is-topic-index" data-topics-browser>
  <p class="topics-browser__lead">Choose a topic to explore its articles.</p>

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
          <a class="topic-collection__open" href="#{{ category_name | slugify }}">View articles <span aria-hidden="true">&rarr;</span></a>
        </div>

        <div class="article-grid">
          {% for post in category_posts %}
          {% assign words = post.content | number_of_words %}
          {% assign minutes = words | plus: 199 | divided_by: 200 %}
          <article class="article-card">
            <a href="{{ post.url | relative_url }}">
              {% if post.image %}
              <div class="article-card__image">
                <img src="{{ post.image | relative_url }}" alt="" width="640" height="400" loading="lazy" decoding="async">
              </div>
              {% endif %}
              <div class="article-card__body">
                <div class="article-card__meta">
                  <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %e, %Y" }}</time>
                  <span aria-hidden="true">&middot;</span><span>{{ minutes }} min</span>
                </div>
                <h3>{{ post.title }}</h3>
                <div class="article-card__excerpt">{{ post.excerpt | strip_html | truncatewords: 20 }}</div>
                <span class="article-card__read">Read article <span aria-hidden="true">&rarr;</span></span>
              </div>
            </a>
          </article>
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
      var requested = decodeURIComponent(window.location.hash.slice(1)).toLowerCase();
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
