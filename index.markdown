---
layout: splash
title: "Machine Learning Spring 2026"
header:
  overlay_color: "#000"
  overlay_filter: "0.4"
  overlay_image: website_graphics/LaneThomasky.png
---

{% include search-box.html %}

## Education Research
[Education Research Information](education_research/education_research)

<!--
## Final project
[Final Project Documents](assignments/assignment17/FinalProject)
-->
## In-class Activities

| Day # | Activity                                                                      |
|-------|-------------------------------------------------------------------------------|
{% for d in (1..8) %}
{%- assign dd = d -%}
{%- if d < 10 -%}{% assign dd = '0' | append: d %}{% endif -%}
{%- assign fname = 'activities/day' | append: dd | append: '.markdown' -%}
{%- assign p = site.pages | where: "path", fname | first -%}

{% if p and p.published == true -%}
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
{%- else -%}
| {{ d }} |  |
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

{% if p and p.published == true %}
{%- comment -%} Build prefixes to remove from the start of the title {%- endcomment -%}
{%- capture pref1 %}Assignment {{ d }}:{% endcapture -%}
{%- capture pref1s %}Assignment {{ d }}: {% endcapture -%}
{%- capture pref2 %}Assignment {{ dd }}:{% endcapture -%}
{%- capture pref2s %}Assignment {{ dd }}: {% endcapture -%}
{%- assign t = p.title | default: p.url -%}
{%- assign t = t | replace_first: pref1s, '' | replace_first: pref1, '' -%}
{%- assign t = t | replace_first: pref2s, '' | replace_first: pref2, '' -%}
{%- assign clean_title = t | strip -%}
{%- if p.no_solutions == true -%}
| {{ p.due_on_class }} | [{{ clean_title }}]({{ p.url | relative_url }}) |
{%- else -%}
| {{ p.due_on_class }} | [{{ clean_title }}]({{ p.url | relative_url }}) ([with show solution button]({{ p.url | relative_url }}?showSolutions=true)) |
{%- endif -%}
{%- endif %}
{% endfor %}


## Other Important Documents
[Notation conventions](assignments/assignment01/notation_conventions)

[Learning as Optimization Key Concepts](assignments/assignment09/LearningAsOptimizationTakeaways)

## Overlay Image Attribution

<span><a href="https://lone-thomasky.de/">Lone Thomasky</a> &amp; <a href="https://bits-und-baeume.org/en">Bits&Bäume</a> / <a href="https://betterimagesofai.org/images?artist=LoneThomasky&title=DigitalSocietyBell"> Digital Society Bell</a> / <a href="https://creativecommons.org/licenses/by/4.0/">Licenced by CC-BY 4.0</a></span>

