---
title: "Assignment 11: Bag of Words, Text Classification, and Word Embeddings"
toc_sticky: true 
toc_h_max: 1
layout: problemset
due_on_class: 13
published: true
---

# Learning Objectives

{% capture content %}
* Learn about bag of words methods for representing text as data
* Use a bag of words methods for text classification
* Learn about the concept of word embeddings and understand them as a form of unsupervised learning
* Understand the pros and cons of word embeddings versus the bag of words approach
* Examine word2vec encodings
{% endcapture %}
{% include learning_objectives.html content=content %}
<!--
{% capture content %}
Choose one of the natural language processing tasks listed above (or substitute one of your own).  Do some research to determine some applications for algorithms that solve the problems listed above.  The distinction here is between problems and how a solution to that problem can be used for some purpose (an application).  Some of these problems may be harder to find information on than others, so do your best.  Aim for a medium length paragraph, 5-6 sentences, for your response.  If you choose an NLP problem not listed above, include a brief description of the problem itself along with the applications you found.
{% endcapture %}
{% include problem_with_parts.html problem=content %}
-->

# Text Classification with Bag of Words

In the video from IBM, there were several examples used to motivate the notion of bag of words for text 
classification. Let's use one of the problems mentioned, sentiment analysis, and apply it to analyzing movie reviews.
We'll be using a fairly old dataset for our analysis, but it is one that is easy to work with and big enough for us 
to learn some important skills about working with text.  The dataset is Stanford's [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). 
Here is a snippet from the README.md file that is included with the dataset.

> Large Movie Review Dataset v1.0
> 
> Overview
> 
> This dataset contains movie reviews along with their associated binary
> sentiment polarity labels. It is intended to serve as a benchmark for
>sentiment classification. This document outlines how the dataset was
> gathered, and how to use the files provided. 
> 
> Dataset
>
> The core dataset contains 50,000 reviews split evenly into 25k train
> and 25k test sets. The overall distribution of labels is balanced (25k
> pos and 25k neg). We also include an additional 50,000 unlabeled
> documents for unsupervised learning. 
>
> In the entire collection, no more than 30 reviews are allowed for any
> given movie because reviews for the same movie tend to have correlated
> ratings. Further, the train and test sets contain a disjoint set of
> movies, so no significant performance is obtained by memorizing
> movie-unique terms and their associated with observed labels.  In the
> labeled train/test sets, a negative review has a score <= 4 out of 10,
> and a positive review has a score >= 7 out of 10. Thus reviews with
> more neutral ratings are not included in the train/test sets. In the
> unsupervised set, reviews of any rating are included and there are an
> even number of reviews > 5 and <= 5.

In the [assignment 11 notebook](https://colab.research.google.com/drive/1ka-K8ovaGaceP3nayL2XZ0NyHU_XZF0g?usp=sharing),
you'll be working with this dataset and implementing your own machine learning system for predicting the sentiment of a
movie review using a bag of words representation.

## Bag of Words and Machine Learning Bias

{% capture problem %}
Let's do a little spiraling back to one of the big ideas in machine learning we started the semester with.  We want to
draw your attention to this specific example.

> You may have heard that [Amazon
> scrapped a secret AI recruiting tool that showed bias against women](https://www.reuters.com/article/us-amazon-com-jobs-automation-insight/amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idUSKCN1MK08G).
More specifically, the tool performed automatic keyword analysis of job applications to predict whether or not the 
> applicant was worth forwarding on to a human for further evaluation. Early in the development of this system 
> researchers discovered that the model the system had learned placed a negative weight on words such as "women's" 
> as well as the names of some women's colleges.

Given what you just learned about the bag of words approach and what we learned about
[confounding variables in assignment 5](../assignment05/assignment05#confounding-variables), how might Amazon's 
system have learned to associate negative feature weights with the gendered words or words associated with women's 
colleges?

{% endcapture %}
{% capture solution %}
The Amazon engineers probably didn't think to screen out particular words from their machine learning model.  Likely,
they assigned a dimension in their bag of words to all unique words as a way to increase the predictive power of the
model.  In the data there was likely a correlation between resumes not doing as well and the presence of gendered words
and the names of women's colleges.  It's hard to say why this correlation might have existed without more 
investigation (e.g., it could have been conscious or subconscious bias on the part of the evaluations that were used 
to make the training set, some systemic factor, or a combination).  Given this correlation, the machine learning 
model associated a negative weight with these words and baked it into the model.  In this way a correlation (that 
having these words in your resume was correlated with being screened out) was made causal by the model (if this 
model were to be applied to real resumes, then people with these words would be more likely to be discriminated 
against).
{% endcapture %}
{% include problem.html problem=problem solution=solution %}


# Word Embeddings

The concept of a word embedding was introduced in the [day 12 materials](../../activities/day12). Word embeddings 
overcome a key limitation with bag of words approaches.  Specifically, in the bag of words approach, each word is 
represented as an *independent* dimension in the vector that represents a particular piece of text.  This means that 
any machine learning task you solve using these vectors needs to learn how the presence or absence of each word in 
the text correlates with the task at hand (no information sharing between words is leveraged).

{% capture problem %}
Before getting into word embeddings in more detail, want to make sure you have a good handle on an important drawback of bag of words approaches.

Suppose, we had a training set consisting of the following movie reviews (you can assume that these are the only reviews in the training set and that we trained the model using a similar technique to what we used in assignment 10).

|--------|-------|
| Review | Label |
|---------|---------|
| The casting of the movie was impeccable | + |
| The movie was great | + |
| The movie was awful | - |
| The movie was the worst I've ever seen | - |
| The movie was an affront to the art of film-making | - |

Explain why a bag of words model trained on this data would have a difficult time evaluating the following movie reviews from a test set.

* "The movie was fantastic"
* "The cast of the movie did a superb job"

{% endcapture %}
{% capture solution %}
For the first review, "the movie was fantastic", the word "fantastic" does not appear in our training set.  Even though fantastic and great are closely related words, in the bag of words approach we treat each word as an independent dimension in our input vector.  If we want to understant fantastic as a synonym for great, we would need training data of movie reviews that contains the word fantastic.

For the second movie review, "The cast of the movie did a superb job", even though we use many similar words to what is present in the training set, the forms of the words (e.g., the chosen tenses) prevent a match with the training set.  In order to generalize to the word forms in this movie review, we would have to have the same word forms represented in the training set.
{% endcapture %}
{% include problem.html problem=problem solution=solution %}

[Word embeddings](https://en.wikipedia.org/wiki/Word_embedding) were introduced as a way to overcome the issues highlighted by the previous problem.  Instead of treating each word as an independent entity, we can learn to represent (embed) each word in a vector space that preserves key properties of the words themselves.  Let's use the symbol $r$ to represent our embedding (we'll use $r$ since it is a *representation* of the word).  We can think of $r$ as a function from words to the vector space $\mathbb{R}^d$ (don't get confused by this notation, $\mathbb{R}^d$ just means a d-dimensional vector of real numbers).

In order to learn our word embedding function $r$, we can use a form of machine learning called [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning).  As we discussed in the previous module, unsupervised learning involves learning from unlabeled data (in contrast to the supervised learning setting we've been studying for most of the term where we assume we have access to a training set consisting of input / output pairs).  We can use the concept of unsupervised learning as a way to create word embeddings.  There are quite a few ways to accomplish this goal, but two foundational approaches were proposed in the paper [Efficient Estimation of Word Representations in Vector Space
](https://arxiv.org/abs/1301.3781).  Here is the key figure from the paper.

{% include figure.html
img="figures/word2vec.jpg"
width="80%"
alt="Two choices for learning word embeddings.  On the left is the continuous bag of words (CBOW) approach where the center word is predicted from the context.  On the right is the skip gram approach where the surrounding words are predicted from the center word."
caption="Given a sequence of words, we can pose a prediction task where we try to either predict the center word based on the embeddings of the surrounding words (CBOW) or the predict the surrounding words based on the center word (skip-gram)." %}
{% assign word2vec = figure_number %}

As mentioned in the caption for {% include figure_reference.html fig_num=word2vec %}, we can use the data itself to pose a prediction task.  You might be wondering how we can call this unsupervised learning given that we are trying to predict something (either the surrounding words or the center word).  Well, the key is that the thing we are trying to predict is derived directly from the data itself (there is no need for any additional information, or label, to be added that is not in the data already).  As such, we can use this approach to learn a word embedding from a database of text (without the need for any additional labeling).

# Word2vec

As mentioned before, word2vec was introduced in the paper [Efficient Estimation of Word Representations in Vector Space
](https://arxiv.org/abs/1301.3781).  We don't think you need to read the paper (but you are certainly welcome to!), but we do want you to get a feel the word embeddings created by word2vec.  We have put together [a notebook that downloads the word embeddings and allows you to explore them a bit](https://colab.research.google.com/github/olinml2024/notebooks/blob/main/ML24_Assignment11.ipynb).

# Bias in Word Embeddings

{% capture problem %}
Depending on what experiments you tried with word2vec, you may have already seen some examples of bias.  We would like you to read the paper [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://www.researchgate.net/profile/Venkatesh-Saligrama/publication/305615978_Man_is_to_Computer_Programmer_as_Woman_is_to_Homemaker_Debiasing_Word_Embeddings/links/57a20cd508aeef8f311e0871/Man-is-to-Computer-Programmer-as-Woman-is-to-Homemaker-Debiasing-Word-Embeddings.pdf).  The paper gets quite technical in places, although many of the ideas you have seen before (PCA??!?).  We would like you to read sections 1-4 of the paper (sadly PCA only shows up in the later sections of the paper).  Please take notes on key takeaways and unanswered questions.  If you'd like to go into the latter sections of the paper (section 5 and beyond), please feel free to do so (this is not required, at all).

It's also probably worth mentioning that the literature on bias in word embeddings is quite extensive with a lot of fascinating things to explore (and we'd love to learn from you if you if you do more explorations!).

{% endcapture %}
{% include problem_with_parts.html problem=problem %}