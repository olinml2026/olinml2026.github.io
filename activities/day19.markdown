---
title: "Day 19: Convolutional Neural Networks" 
toc_sticky: true 
toc_h_max: 1
layout: problemset
published: false
---

> still finalizing

{% capture agenda %}
* 3:45-3:55pm: Quick ConvNet review
* 3:55-4:15pm: Image filter debrief
* 4:15-4:35: Data augmentation
* 4:35-5:15: Privacy in machine learning
* 5:15-5:25pm: Next assignment preview
{% endcapture %}
{% include agenda.html content=agenda %}


# Overview of a ConvNet/CNN/Convolutional Neural Network

In your assignment, you looked at some of:
* This [interactive visual overview of CNNs from a collaboration between Georgia Tech and Oregon State](https://poloclub.github.io/cnn-explainer/){:target="_blank"}. This one will allow you to explore each of the layers and functions. You can click on each of the parts to see more. There's a little video at the end that shows how to use the tool. 
* This [write-up with some helpful visualizations by Ujjwal Karn](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets){:target="_blank"}.
* [One of the earlier types of these visualizations focused on handwritten numbers](https://adamharley.com/nn_vis/){:target="_blank"}  by Adam Harley.
* [Training on MNIST in the browser by Karpathy](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html){:target="_blank"}. This one shows the weights and the gradients.

You might have some questions, like:  
* I looked at the architecture, but I'm not sure if I could explain in. Can you help?
* Why not do this whole thing as a bunch of fully connected layer?
* Everyone loves to make these brain analogies, is this really what the brain does?

# Image filter debrief

Filters: Not just to keep you from saying something you'll regret. They also help ConvNets process images!

In your assignment, you manually created filters to detect different properties of images (e.g., vertical lines). There are many correct ways to do this, and they may lead to different results. At tables, compare your filters and results with others. Be prepared to share one observation or comparison with the larger group. 

# Data augmentation

TODO (bring back some of the image transform stuff from 2024)

# Privacy in machine learning

TODO

# Next assignment preview
We'll discuss the next assignment and get started.
