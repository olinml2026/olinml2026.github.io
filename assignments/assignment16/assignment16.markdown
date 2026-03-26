---
title: "Assignment 16: SBERT and Finetuning"
toc_sticky: true 
toc_h_max: 1
layout: problemset
due_on_class: 18
published: true
---

# Learning Objectives

{% capture content %}
* Learn about SBERT models
* Learn about the idea of finetuning and apply this procedure to a dataset and model of your choosing
{% endcapture %}
{% include learning_objectives.html content=content %}

{% capture notice %}
It will help a lot to complete this assignment if you have Google Colab pro.  As mentioned early in the class, you 
can get free usage of Colab pro for being a student  (see this [link on Canvas for instructions](https://olin.
instructure.com/courses/1000/pages/get-colab-pro?module_item_id=20148)).  If you are not able to gain access to 
Colab pro for any reason, please talk to me, and we'll see if we can find another way forward.
{% endcapture %}
{% include notice.html content=notice %}

# SBERT

SBERT ([Sentence-Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)) is a hugely influential paper that forms the basis of many modern approaches for computing similarity between
two pieces of text.  As we saw in the previous assignment, being able to evaluate the similarity between two pieces of text can be used for tasks 
such as search and question answering.  The main ideas of SBERT are described in [day 17](../../activities/day17).  
You can also learn about SBERT from the original paper (linked previously in this paragraph).  You can also find 
more information using the following resources.

# Finetuning

{% capture problem %}
We've put together [a notebook with some starter code](https://colab.research.google.com/drive/1jipb75ZcHhMNbMDKY8JUnshnHC1oIFo-?usp=sharing) that will allow you to finetune a sentence embedding model on a 
dataset of your choosing.  You will be building off of this code to try your own experiment.
{% endcapture %}
{% include problem.html problem=problem %}