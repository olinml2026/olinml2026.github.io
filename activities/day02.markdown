---
title: "Day 2:  ML as Optimization and Linear Regression"
toc_sticky: true 
toc_h_max: 1
layout: problemset
published: true
---

{% capture agenda %}
* 3:45-3:50pm: Welcome to online class! Chelsea will say hi.
* 3:50-4:05pm: Debrief on previous assignment
* 4:05-4:15pm: ML as Optimization Key Ideas and Supervised Learning Problem Setup
* 4:15-5:25pm: Work on assignment collaboratively
{% endcapture %}

{% include agenda.html content=agenda %}

# Debrief on Assignment 1

Let's go into breakout rooms and do a quick debrief on assignment 1.
* Share you responses to exercise 2.
* Note any lingering confusions.

# Learning as Optimization Key Terminology

While it hasn't always been the case, optimization is at the core of modern machine learning.  If you aren't 
familiar with the idea of optimization, we will be building some intuition together.  An optimization problem 
typically consists of the following key ingredients, which together define a vocabulary that helps us see 
connections between seemingly different problems.

Optimization has a lot of complex sounding vocabulary / jargon.  One of my goals in this unit is to help you feel 
comfortable with this jargon and connect it to concrete concepts in a number of real world examples.  Let's start to 
unpack some of these ideas together.  It may seem confusing at first, but we'll get there together.

* *Decision variables:* what is the "space" of candidate solutions?  Each variable may represent some knob that 
  controls something about the solution.
* *Object function:* what are we trying to achieve?  What does it mean for a particular candidate solution to be good 
  or bad?
* *Constraints:* sometimes we are able to pick any values for our decision variables such that the objective 
  function is maximized (unconstrained optimization).  Sometimes, there are specific solutions that are disallowed 
  (infeasible).  The rules that govern which solutions are feasible (allowed) versus infeasible (disallowed) are 
  called constraints.  Optimization problems with constraints are known as "constrained optimization problems"

One thing that is important to realize about the concepts above is that they help you frame an optimization problem, 
however, they do not, by themselves, tell you how you would actually solve the optimization problem.  Coming up with 
a solution to an optimization problem is the domain of optimization algorithms (which we will learn about soon). 

Let's think through a problem you've seen before in QEA: smile detection.  We'll go over this as a group.

# The Supervised Learning Problem Setup (Learning as Optimization)

We're now switching gears to talk about how machine learning can be thought of as an optimization problem.  We're going to start with a mathematical definition the simplest type of machine learning: supervised learning.  Along the way you'll get a chance to build your conceptual knowledge about how learning can be thought of as a learning problem.

> Note: this next section is also in the homework, but we wanted to have a chance to go over this together.

Suppose you are given a set of training data points, $(\mathbf{x_1}, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)$ where each $\mathbf{x_i}$ represents an element of an input space (e.g., a d-dimensional feature vector) and each $y_i$ represents an element of an output space (e.g., a scalar target value).  In the supervised learning setting, your goal is to determine a function $\hat{f}$ that maps from the input space to the output space.  For example, if we provide an input $\mathbf{x}$ to $\hat{f}$ it would generate the predicted output $\hat{y} = \hat{f}(\mathbf{x})$.

We typically also assume that there is some loss function, $\ell$, that determines the amount of loss that a particular prediction $\hat{y_i}$ incurs due to a mismatch with the actual output $y_i$.  We can define the best possible model, $\hat{f}^\star$ as the one that minimizes these losses over the training set.  This notion can be expressed with the following equation  (note: that $\argmin$ in the equation below just means the value that minimizes the expression inside of the $\argmin$, e.g., $\argmin_{x} (x - 2)^2 = 2$, whereas $\min_{x} (x-2)^2 = 0$).

$$
\begin{aligned}
\hat{f}^\star &= \argmin_{\hat{f}} \sum_{i=1}^n \ell \left ( \hat{f}(\mathbf{x_i}), y_i \right )
\end{aligned} 
$$

# Getting Started on Linear Regression

A particular type of supervised learning problem is called linear regression or least squares.  You met this algorithm way back in QEA1, but we don't expect you to recall all of those details!  We're going to go over linear regression from a different perspective in this class.  We find that it often takes us multiple encounters with the same idea to start to really achieve proficiency (hopefully this is another step along that journey for you).

The way we recommend engaging with this material is by [starting on assignment 2](../assignments/assignment02/assignment02).  We'll open up some breakout rooms and you can join them as you like.