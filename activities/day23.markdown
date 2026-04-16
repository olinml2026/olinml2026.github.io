---
title: "Day 23: Energy Usage of LLMs and Project Worktime"
toc_sticky: true 
toc_h_max: 1
layout: problemset
published: true
---

{% capture agenda %}
* 3:45-4:25: Environmental Impacts of Large Language Models
* 4:25-5:25pm: Project worktime
{% endcapture %}
{% include agenda.html content=agenda %}

# Environmental Impacts of Large Language Models

{% capture problem %}
The environmental impact of large language models is one of the biggest concerns around the technology.  Here are a 
few resources you can check out to learn more.

As you go, you engage with these resources try to think critically about what you are reading.
* Who is writing the article?
* How are they measuring environmental impact?  Is it accounting for inference-only or inference with amortized 
  training costs?
* What costs might be left out of their analysis?

* [A measurement of Google Gemini's energy usage per query](https://cloud.google.com/blog/products/infrastructure/measuring-the-environmental-impact-of-ai-inference)
* [The environmental impact of using a particular GPU in a particular datacenter](https://mlco2.github.io/impact/)
* Some models on Hugging Face actually show training CO2 emissions.  The [JobBERTv3 model](https://huggingface.co/TechWolf/JobBERT-v3) uses comparatively little CO2 (0.717 kg) whereas the [BLOOM model](https://huggingface.co/bigscience/bloom/edit/main/README.md) uses a lot (24,700 kg).
* [Gen AI’s Environmental Ledger: A Closer Look at the Carbon Footprint of ChatGPT](https://piktochart.com/blog/carbon-footprint-of-chatgpt/)
* Find some articles or estimates of your own.  What is the environmental impact a particular AI model?  Perhaps 
  focus on models that may be either more or less efficient.

We'll do a quick report out on your findings.
{% endcapture %}
{% include problem.html problem=problem %}


Here is [a notebook](https://colab.research.google.com/drive/1-JjJgMVL3MlmJNTi4fDRdP_ucv5FjGV2?usp=sharing) that will allow you to measure the energy usage of sentence transformer models.  Remember that 
this is the same technology that is used in the EchoMinds app (and we learned about the SBERT model that forms the 
basis of these approaches).

{% capture problem %}
Using the notebook above, estimate the carbon footprint of using the EchoMinds app.  You will have to make some 
assumptions here, but hopefully you can get a rough sense.  If you don't know if an assumption is reasonable, just ask!
{% endcapture %}
{% include problem.html problem=problem %}


# Project work time

Please use this time to work on projects. Note that there is a standup next class, which is a week from today.