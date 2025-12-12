---
title: "Day 4: 	Metrics and Meeting ML as Optimization"
toc_sticky: true 
toc_h_max: 1
layout: problemset
published: true
---

{% capture agenda %}
* 10:20-10:45am: Applying Frameworks for Community-Centered ML
* 10:45-10:55am: Supervised Learning Problem Setup
* 10:55-12:00pm: Start assignment 
{% endcapture %}

{% include agenda.html content=agenda %}




# The Supervised Learning Problem Setup (Learning as Optimization)

We're now switching gears to talk about how machine learning can be thought of as an optimization problem.  We're going to start with a mathematical definition the simplest type of machine learning: supervised learning.  Along the way you'll get a chance to build your conceptual knowledge about how learning can be thought of as a learning problem.

> Note: this next section is also in the homework, but we wanted to have a chance to go over this together.

Suppose you are given a set of training data points, $(\mathbf{x_1}, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)$ where each $\mathbf{x_i}$ represents an element of an input space (e.g., a d-dimensional feature vector) and each $y_i$ represents an element of an output space (e.g., a scalar target value).  In the supervised learning setting, your goal is to determine a function $\hat{f}$ that maps from the input space to the output space.  For example, if we provide an input $\mathbf{x}$ to $\hat{f}$ it would generate the predicted output $\hat{y} = \hat{f}(\mathbf{x})$.

We typically also assume that there is some loss function, $\ell$, that determines the amount of loss that a particular prediction $\hat{y_i}$ incurs due to a mismatch with the actual output $y_i$.  We can define the best possible model, $\hat{f}^\star$ as the one that minimizes these losses over the training set.  This notion can be expressed with the following equation  (note: that $\argmin$ in the equation below just means the value that minimizes the expression inside of the $\argmin$, e.g., $\argmin_{x} (x - 2)^2 = 2$, whereas $\min_{x} (x-2)^2 = 0$).

\begin{align}
\hat{f}^\star &= \argmin_{\hat{f}} \sum_{i=1}^n \ell \left ( \hat{f}(\mathbf{x_i}), y_i \right )
\end{align} 

# Getting Started on Linear Regression

A particular type of supervised learning problem is called linear regression or least squares.  You met this algorithm way back in QEA1, but we don't expect you to recall all of those details!  We're going to go over linear regression from a different perspective in this class.  We find that it often takes us multiple encounters with the same idea to start to really achieve proficiency (hopefully this is another step along that journey for you).

The way we recommend engaging with this material is by [starting on assignment 3](../assignments/assignment03/assignment03).  We would like to invite those who want to adopt a quieter, more independent work style to travel over to MAC126.  We'll circulate around to that room and answer any questions.  Those who want to collaborate and work through problems together, can remain here.