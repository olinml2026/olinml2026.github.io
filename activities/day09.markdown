---
title: "Day 9: Validation sets, project planning, and multilayer perceptron"
toc_sticky: true 
toc_h_max: 1
layout: problemset
published: true
---

{% capture agenda %}
Asynchronous class due to snowday.
{% endcapture %}
{% include agenda.html content=agenda %}

Watch this [video on train, test, and validation sets](https://www.youtube.com/watch?v=dSCFk168vmo). While we have 
covered train and test set, we haven’t covered validation sets.  When the video creator talks about hyperparameters, 
these are settings for your machine learning algorithm that are not explicitly fit to the training data (they are 
not part of the gradient descent optimization process). We haven’t learned about many of these hyperparameters yet, 
but we will learn about more soon.  One example of a hyperparameter is the value of lambda in ridge regression (this 
is the amount we penalize error in fitting the training data versus the penalty for large values in our weights).  
In this case, the training data would tune the weights of the linear regression, the validation set would let us 
pick the best value of lambda, and the test set would give us an unbiased estimate of performance.

Go through the [day 9 notebook](https://colab.research.google.com/github/olinml2024/notebooks/blob/main/ML24_Day09.ipynb). There are three things we want you to get out of this notebook.

Pytorch is quite similar in its basic concepts to the micrograd framework you implemented.
We can use pytorch to compute a line of best fit. This will allow us to visualize the optimization process more easily.
We can use pytorch modules (e.g., nn.Linear) to make our lives easier.
Read the [small data mini project assignment](https://olinml2024.github.io/assignments/assignment09/assignment09), 
which we will start on Thursday. I will update the assignment to reflect a shorter duration this year (with a 
corresponding reduction in scope). Last year we did this in 3 class sessions, and this year we will aim for 2 
sessions. By the end of our asynchronous class session you should have at least one concrete idea for a project 
(meaning an application and a potential dataset). Please populate your current thinking into this shared
[Google doc](https://docs.google.com/document/d/11rHezUutorn5D6OqLT1PfIPUu083lT4M6ZhNtmjelO8/edit?usp=drivesdk).

(As time allows) Start [the next assignment](../assignments/assignment09/assignment09).  I highly recommend that you 
watch the videos in the external resources section (since we are moving this material into a class session, the videos will provide important framing)