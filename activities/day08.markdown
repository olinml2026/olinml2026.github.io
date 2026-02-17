---
title: "Day 8: From Micrograd to Pytorch and Starting COMPAS"
toc_sticky: true 
toc_h_max: 1
layout: problemset
published: true
---

{% capture agenda %}
* 3:45-4:05pm: Instructor-led debrief of the homework... what just happened?!?
* 4:05-4:25am: Cross-entropy loss
* 4:25-5:15pm: From micrograd to Pytorch
* 5:15-5:25pm: Preview of what's to come
{% endcapture %}
{% include agenda.html content=agenda %}

# Instructor-led Debrief

We'll debrief on what happened in the previous assignment.  The focus will be on connecting mathematical concepts to Python.  We hope that by the end of this everything is coming into focus for you (it may take a little longer to fully click).

# From Micrograd to Pytorch

While it may be tempting to ride our micrograd framework for the rest of the semester, you can probably tell that there are some good reasons to move to something *a little* more powerful.  We're going to be using the `pytorch` framework for the remainder of the scaffolded work in this course (it's possible you might venture into a different framework for the final project).  Machine learning frameworks like `pytorch` provide some really important capabilities for us.

* An autograd engine
* Built-in optimizers (that do, for example, gradient descent)
* Optimized code that can efficiently handle large models (e.g., by running on a GPU or across several GPUs)
* Specific building blocks for machine learning algorithms that are used by current state of the art algorithms.
* The ability to be extended easily when the library doesn't provide the necessary functionality.

To help introduce `pytorch`, we're going to jump right into a looking at some `pytorch` code.  This is a great chance to practice reading code and looking up documentation.  Your goal should be to understand the given code as well as possible.  If there are pieces that you can't figure out, please ask us or make a note of your confusion so you can revisit it later.  You'll also get a head start on the assignment (so that is a bonus!).

The code in question is in the [assignment 8, part 2 Colab notebook](https://colab.research.google.com/github/olinml2024/notebooks/blob/main/ML24_Assignment08_part_2.ipynb). The first two code cells load a dataset of handwritten digits and visualize them.  The third code cell is where the action is, we'd like you to go over that one, read documentation, ask ChatGPT, ask an instructor, etc., so that you leave here today with a solid understanding of a training / testing loop in `pytorch`.

# More Resources on Pytorch

We're going to be introducing Pytorch functionality on an as needed basis, but if you'd like to get some more practice with the basics, we recommend checking out some of [the Pytorch tutorials](https://pytorch.org/tutorials/).  Start with the [basics of using Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html).

# Preview of Where We Are Going

* As part of the next assignment, you'll be doing your first quality-assessed deliverable.  This will go out on 
  The deliverable will be open-Internet, closed GenAI, and done individually.  The assignment will cover concepts of 
  model evaluation and metrics.
* Next class, we'll be discussing the COMPAS algorithm for recidivism prediction, which is a famous case study on 
  bias in machine learning systems.  The topics discussed are quite sensitive, and we ask that you approach the 
  readings and class discussion with an open mind (see [this section of the next assignment](../assignments/assignment08/assignment08#compas-model-race-criminal-justice-and-machine-learning) for some specifics on 
  this).  On Monday, Paul will provide some discussion guidelines to help 
  guide class time.