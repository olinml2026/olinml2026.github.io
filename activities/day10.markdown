---
title: "Day 10: COMPAS discussion"
toc_sticky: true 
toc_h_max: 1
layout: problemset
published: true
---

{% capture agenda %}
* 3:45-3:55: Instructor-led debrief on assignment 9.
* 3:55-4:15pm: Gathering information based on sources
* 4:15-4:30pm: Large group debrief and introduction of impossibility theorem 
* 4:30-4:50: Small group exploration and sense-making of fairness metrics
* 4:50-5:25pm: Large group discussion of the big picture
{% endcapture %}

{% include agenda.html content=agenda %}

# Debrief on Assignment 9

I'll go over some key points from assignment 9.

Let's start by reviewing the first notebook from assignment 9.  We want to make sure you understand the key idea of 
hand-coded features versus learning features from data.

Here are the loss graphs (cross entropy) that you generated for the hand-written digit dataset.

{% include figure.html
img="images/learning_curve_ce.png"
width="100%"
alt="A graph of training and test cross entropy as a function of gradient descent step.  The curves begin near 2.4 and settle around 1.7"
caption="The cross entropy on the handwritten digit classification task.  The x-axis refers to the number of gradient descent steps." %}

Here is the equation for cross entropy.
$$
\begin{aligned}
ce(\hat{\mathbf{y}}, y) = \sum_{i=1}^{k} -\mathbf{I}[y = i] \log \hat{y}_i
\end{aligned}
$$

Let's interpret this together.

# Gather information based on external sources

On the whiteboard, please gather pieces of information that we learned from the readings or other sources. Take 
turns having each person share one thing that they wrote down for Exercise 1 in the last assignment.

# Large group debrief and introduction of the impossibility theorem 

Here are some
[summary slides](https://docs.google.com/presentation/d/1I1rMkqJYiOuaYAmzKzj3oshUxqdPa_wiOlqtnt4qbLc/edit?
usp=sharing), which also include reference to the study Sam mentioned with humans attempting to predict re-arrest.

We'll summarize some key takeaways and show an example from an extreme version to help us wrap our heads around 
different models of fairness. [https://medium.com/@alex.liu.
roc/understanding-the-impossibility-of-fairness-199bba6c9072](https://medium.com/@alex.liu.roc/understanding-the-impossibility-of-fairness-199bba6c9072)

# Small group exploration and sense-making of fairness metrics 

The field of fairness and applications to human or algorithmic decision making is vast. Here are a few resources to guide your exploration of fairness metrics:
* [IBM's exploration of COMPAS and fairness metrics](https://aif360.res.ibm.com/data)
* [One group that made a tool for fairness and has a flowchart](https://www.datasciencepublicpolicy.org/our-work/tools-guides/aequitas/)
* [Orange and blue dot example of fairness from the last assignment](https://research.google.com/bigpicture/attacking-discrimination-in-ml/)
* [Fair prediction with disparate impact - math paper on COMPAS](https://www.andrew.cmu.edu/user/achoulde/files/disparate_impact.pdf)

# Large group discussion of the big picture

We will close our computers for this part and have a large group guided discussion. 
