---
title: "Day 16: Trust and Trustworthiness in Machine Learning and Starting BERT"
toc_sticky: true 
toc_h_max: 1
layout: problemset
published: true
---

{% capture agenda %}
* 3:45-4:35pm: Discussion of Trust and Trustworthiness in Machine Learning Systems
* 4:35-5:25pm: Introduction to BERT
{% endcapture %}
{% include agenda.html content=agenda %}

> Note: Dr. Chelsea Andrews will be here today to observe class and get a feel for what the day-to-day rhythm in 
> Machine Learning is like.  She will not be recorded what you say, and this class session is not part of the 
> research study we are doing on the class.

# Discussion on Trust and Trustworthiness Paper

Let's summarize the main points from the paper in small groups.  Here are some prompts to get you going, but we 
recommend you raise the main points that you uncovered as well.
* Define key terms (e.g., how do the authors define trust, trustworthiness, contractual trust, etc.)
* How are trust and trustworthiness different?  When could a model be trustworthy and not trusted or trusted and not 
  trustworthy?  How does this relate to figure 6?

In figure 3, the authors propose a three parameter model for trust in AI systems.  Do each of these terms make sense?
For your own interactions with AI, do you agree that these parameters are associated with trust in AI systems?
* Parameter one: Benevolence.  Social Responsibility, Ethical Behavior, Sustainability, Fairness 
* Parameter two: Integrity.  Standards and Guidelines, Certifications, Government Regulations
* Parameter three: Ability.  Technical correctness, transparency, explainability, reliability, privacy and data 
  governance, technical robustness and safety.

In figure 5, the authors propose a framework for distrust makers in AI systems.  Take some time to go through the 
terms in the figure and make sure you understand them.

We're revisiting the EchoMinds case study as an entry point into the issues around Trust and Trustworthiness in 
machine learning systems.  Here are some relevant examples from the summer work.

Analyze the following case studies using the frameworks in figure 3 and figure 5 of the paper. That is, for each case 
study analyze what aspects of the system's design or behavior increase or decrease trust.
* [Ashley's anecdote of the positive COVID test](https://olincollege.sharepoint.com/:v:/s/EducateAI/IQC_dRtNEQL3QJj78LLZV-PGAXDSS_DK_NaW4m3Z47qKwyA)
* ![A Facebook post detailing an experience with having an AI app flag a bill as $9](images/ninedollar.png)
* [Meta Glasses: Total Game-Changer for Me—Anyone Else?](https://www.reddit.com/r/Blind/comments/1nqw16e/meta_glasses_total_gamechanger_for_meanyone_else/)
* The EchoMinds app.  Some key features to keep in mind (for the app in final version): data is stored locally on the 
  device, the app doesn't use generative AI (only content entered into the app is returned), the app analyzes data 
  locally and uses little power.

{% capture problem %}
Come up with your own case study and evaluate it with respect to the parameters of trust.  We'll do a report out to 
the whole class towards the end of our discussion time.

If there's time, using the framework in figure 5, analyze a case study where an AI system has engendered distrust.  
That is, can you think of a system that potential intersects with the elements in figure 5, and explain how the 
system's design or behavior may have contributed to mistrust.
{% endcapture %}
{% include problem.html problem=problem %}


# Introduction to BERT

We're going to close out our text as data module by learning about how transformer architectures can be used in 
defining text similarity.  We'll show how this capability is at the core of the EchoMinds app, and you'll get a 
chance to implement a series of models that will get increasingly good performance.

I'll talk a little bit about BERT and draw some things on the chalkboard.  You'll also have some time to start the 
assignment.