---
layout: page
title: Search
permalink: /search/
---

<style>
    #results-container a,
    #results-container a:visited {
        color: #5dade2; /* light blue */
        text-decoration: none;
    }
    #results-container a:hover {
        color: #3498db; /* slightly darker blue on hover */
        text-decoration: underline;
    }
    #results-container a h1,
    #results-container a:visited h1 {
        color: #5dade2; /* light blue */
    }
    
    #results-container a:hover h1 {
        color: #3498db; /* darker blue on hover */
    }

</style>

<div id="search-container">
    <input type="text" id="search-input" placeholder="Search through the blog posts...">
    <ul id="results-container"></ul>
</div>

<script src="{{ site.baseurl }}/assets/simple-jekyll-search.min.js" type="text/javascript"></script>

<script>
    SimpleJekyllSearch({
    searchInput: document.getElementById('search-input'),
    resultsContainer: document.getElementById('results-container'),
    searchResultTemplate: '<div style="text-align: left !important;"><a href="{url}"><h1 style="text-align:left !important;">{title}</h1></a><span style="text-align:left !important;">{date}</span></div>',
    json: '{{ site.baseurl }}/search.json'
    });
</script>
