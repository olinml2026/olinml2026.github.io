---
title: Assignment 4
toc_sticky: true 
toc_h_max: 1
layout: problemset
due_on_class: 5
published: true
no_solutions: true
---

# Learning Objectives

{% capture content %}
* Evaluate an Information Retrieval System
* Combine Qualitative and Quantitative Methods for System Evaluation
{% endcapture %}
{% include learning_objectives.html content=content %}

# Model Evaluation for Retrieval Systems

{% capture content %}
Read this [article about evaluating information retrieval systems](https://www.pinecone.io/learn/offline-evaluation/).  You can stop reading at the section called NDCG@K
since the data we will use for evaluation doesn't match the requirements needed to apply this metric.
{% endcapture %}
{% include external_resources.html content=content %}


# Evaluating Machine Learning Systems

{% capture content %}
Come up with a plan for evaluating the EchoMinds app.  You may consider evaluating the user experience of the system 
(e.g., do people find the app helpful, usable, etc.) itself, testing the underlying machine learning system (e.g., 
by evaluating it on a dataset), or some combination.  You may also find it
useful to consider [qualitative versus quantitative methods of assessment](https://medium.com/@sujathamudadla1213/difference-between-quantitative-and-qualitative-evaluation-of-models-127d7e51da56)
or [aspects of the user experience](https://medium.com/design-bootcamp/introducing-the-ai-ux-self-assessment-questionnaire-99b67eadac4a).

You can use bullet points or paragraphs to describe your plan, but you should put some real thought into this.  Try 
to spend at least 45 minutes thinking through this problem.
{% endcapture %}
{% include problem.html problem=content %}

# Model Analysis

{% capture content %}
Go through the [Colab notebook on analyzing the machine learning model used in the EchoMinds app](https://colab.research.google.com/drive/1R2O3-70qcj7TU7iRyPOzjmiISFUas3YR?usp=sharing).
{% endcapture %}
{% include problem.html problem=content %}

# (Optional) Download the Echo Minds App

If you didn't do this in class and you want to play around with the app yourself, here are the instructions to get the 
EchoMinds app on your iOS device. If you do not have an iOS device, please consider coming to my office hours this Wednesday from 1-2pm (MAC212).  You can test 
it on one of the devices in my lab.

1. Download the TestFlight app from the App Store
2. Click the following link (on your iPhone or iPad) to [join the EchoMinds beta](https://testflight.apple.com/join/hMXSxFrp).
3. Load some notes about accessible tech (find this in the settings menu)