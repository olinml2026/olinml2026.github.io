---
layout: splash
title: "Machine Learning Spring 2026"
header:
  overlay_color: "#000"
  overlay_filter: "0.4"
---

{% include search-box.html %}

## Final project
[Final Project Documents](assignments/assignment17/FinalProject)

## In-class Activities

[Sample solutions for in-class assignments](https://github.com/OlinDSA2025/SampleSolutions) will be made available on GitHub.

| Day # | Activity                                                                      |
|-------|-------------------------------------------------------------------------------|
{% for d in (1..26) %}
{%- assign dd = d -%}
{%- if d < 10 -%}{% assign dd = '0' | append: d %}{% endif -%}
{%- assign fname = 'activities/day' | append: dd | append: '.markdown' -%}
{%- assign p = site.pages | where: "path", fname | first -%}

{% if p and p.published != false -%}
{%- comment -%} Build prefixes to remove from the start of the title {%- endcomment -%}
{%- capture pref1 %}Day {{ d }}:{% endcapture -%}
{%- capture pref1s %}Day {{ d }}: {% endcapture -%}
{%- capture pref2 %}Day {{ dd }}:{% endcapture -%}
{%- capture pref2s %}Day {{ dd }}: {% endcapture -%}
{%- assign t = p.title | default: p.url -%}
{%- assign t = t | replace_first: pref1s, '' | replace_first: pref1, '' -%}
{%- assign t = t | replace_first: pref2s, '' | replace_first: pref2, '' -%}
{%- assign clean_title = t | strip -%}
| {{ d }} | [{{ clean_title }}]({{ p.url | relative_url }}) |
{%- endif %}
{% endfor %}

##  Assignments

| Due at beginning of class # | Assignment |
|-------|------------|
{% for d in (1..20) %}
{%- assign dd = d -%}
{%- if d < 10 -%}{% assign dd = '0' | append: d %}{% endif -%}
{%- assign fname = 'assignments/assignment' | append: dd | append: '/assignment' | append: dd | append: '.markdown' -%}
{%- assign p = site.pages | where: "path", fname | first -%}

{% if p and p.published != false -%}
{%- comment -%} Build prefixes to remove from the start of the title {%- endcomment -%}
{%- capture pref1 %}Assignment {{ d }}:{% endcapture -%}
{%- capture pref1s %}Assignment {{ d }}: {% endcapture -%}
{%- capture pref2 %}Assignment {{ dd }}:{% endcapture -%}
{%- capture pref2s %}Assignment {{ dd }}: {% endcapture -%}
{%- assign t = p.title | default: p.url -%}
{%- assign t = t | replace_first: pref1s, '' | replace_first: pref1, '' -%}
{%- assign t = t | replace_first: pref2s, '' | replace_first: pref2, '' -%}
{%- assign clean_title = t | strip -%}
| {{ p.due_on_class }} | [{{ clean_title }}]({{ p.url | relative_url }}) |
{%- endif %}
{% endfor %}


## Other Important Documents
[Notation conventions](assignments/assignment01/notation_conventions)

[Learning as Optimization Key Concepts](assignments/assignment09/LearningAsOptimizationTakeaways)