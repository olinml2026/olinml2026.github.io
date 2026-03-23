---
title: "Assignment 15: Transformers for Text Embedding"
toc_sticky: true 
toc_h_max: 1
layout: problemset
due_on_class: 17
published: true
---

# Learning Objectives

{% capture content %}
* Articulate the weaknesses of traditional word embedding models like word2vec
* Understand how transformers can be used to improve word embeddings
* Extend word embeddings to embed text with multiple words
{% endcapture %}
{% include learning_objectives.html content=content %}


# Intro

While we used the idea of text generation as a way to motivate the transformer, the transformer architecture is used 
in many other contexts.  As was mentioned in some of the sources we've consulted, the paper ["Attention is All You 
Need"](https://dl.acm.org/doi/pdf/10.5555/3295222.3295349), which introduced the transformer architecture, contained 
results on the problem of machine translation (translating text from one language to another).

In the next two assignments, we're going to revisit the problem of text similarity and information retrieval.  You 
will recall, that being able to retrieve relevant pieces of text from a query is the core computational problem that 
lies at the heart of the EchoMinds app.


<img alt="A screenshot of the EchoMinds app showing notes on various access technology"
src="figures/echominds_screenshot.png" style="float: right; width: 250px; margin: 0 0 1rem 1rem;"/>
{% capture content %}

Remember that the Echominds app was developed in the summer of 2025 as a notetaking tool for the blind and low 
vision users.  The app's core value proposition is to enable notes to be capture easily (either through voice or 
text input) and then retrieved using natural language queries.  The design process that was used to arrive at the 
EchoMinds app was explored in [assignment 3](../assignment03/assignment03.markdown).

The team chose to design the app using a retrieval approach (returning the content the user has explicitly entered 
or imported into the app) rather than a generative approach (e.g., ChatGPT) because 
in their community-engaged design process, the team heard form users that they had been lied to by GenAI systems in the 
past.  By returning only the data that the user had directly put into the system, the user would have confidence in the
accuracy of the results.  This decision closely relates to the topic of trust and trustworthiness of Machine 
learning systems that we explored in [assignment 13](../assignment13/assignment13.markdown).

As an example of the usage model for the EchoMinds app, consider the screenshot on the right, which shows the 
EchoMinds app with many preloading accessibility technology facts.  A student who has recently lost their vision and 
is retraining to do their previous job non-visually, may be learning how to use the JAWS screenreader.  They may 
want to know how to navigate a list of links using JAWS.  They may pose the question "How do I navigate links 
quickly in JAWS?" and the EchoMinds app would search through the accessible technology facts that have been stored 
in the app and return: "Press INSERT+F7 to display a list of all links on the page."
{% endcapture %}

{% include notice.html content=content %}

Over the next two assignment, we'll learn about a series of progressively more powerful approaches to solving the 
task of text similarity and information retrieval.  We'll start from a very simple approach based on word2vec, 
add in the idea of transformers through an approach called BERT (Bidirectional Encoder Representations from 
Transformers), learn how to move from word 
embeddings to sentence embeddings using an approach called SBERT (Sentence BERT), and finally learn how we can 
fine-tune a sentence embedding on our own dataset to squeeze out more performance.

# Initial Approach and Problem Statement

Let's formalize the problem we're trying to solve in the creation of the EchoMinds app.  Given a series of notes 
$x_1, x_2, \ldots, x_n$ where each $x_i$ is some piece of text, given query text $x_q$, return a sorted list of 
notes that are most similar to the query.  The notion of similarity is intentional ambiguous here, but you might 
imagine that the idea of similarity should conform the user's expectations and the structure of natural language.

As an example, let's say we have the following notes.
1. To silence JAWS speech immediately press the Control key.
2. To read the current time in JAWS press Insert + F12.
3. To silence TalkBack speech tap the screen with two fingers once.
4. Moovit is often used by blind travelers for accessible public transit planning.

Let's say we have a query term **"How do I stop JAWS from talking?"**.  Given a particular algorithm for defining 
similarity, we might return the notes in the following order (from most to 
least similar): (1), (3), (2), (4).

To start ourselves off, you'll think through a very simple model for text similarity.

{% capture problem %}
Let's consider an approach to text similarity based on text embeddings.  Let's define a text embedding function 
$\bm{\phi}(x)$ where $x$ is some piece of text and $\bm{\phi}(x)$ returns a vector of numbers in a 
$d$-dimensional space.  We
assume that the directions of the vectors $\bm{\phi}(x)$ in this space captures something about the meaning of the 
text.Given a query $x_q$ and a potential item to retrieve $x_i$ we define the similarity between the pieces of text using
[cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

<div>
$$
\text{cosine\_similarity}(x_q, x_i) = \frac{\bm{\phi}(x)^\top \bm{\phi}(x)}{\|\bm{\phi}(x)\| \|\bm{\phi}(x_q)\|}
$$
</div>

While this formula might look daunting it is nothing more than $\cos(\theta)$ where $\theta$ is the angle between
the vectors $\bm{\phi}(x_q)$ and $\bm{\phi}(x_i)$ (to understand why this is the case, you just need to 
recall 
some  facts about the dot product).

Recall that the Continuous Bag of Words model (CBOW), we assign each word in our vocabulary to a $300$-dimensional 
vector.  Assuming you have a trained version of the CBOW model on hand, propose a method for using the CBOW model to 
compute an embedding of a piece of text.  If you can come up with several approaches, list them here.  Be as 
creative as you like.  For each of the approaches you come up with, describe the key limitations of the approach for 
capturing the underlying meaning of the piece of text (e.g., what are some things your approach would not consider 
about the piece of text that might be important for understanding its meaning).
{% endcapture %}
{% capture solution %}
A simple approach would be to take a piece of text and segment it into a bunch of words $w_1, \ldots w_k$.  For each 
word, we could compute its embedding using word2vec (yielding a sequence of 300-dimensional vectors).  As a first 
pass, we can add these 300-dimensional vectors up to capture the overall meaning of the piece of text.

This approach would fail to encode the order of the words in the text.  It would also fail to model how a specific 
word modifies the meaning of another word (e.g., the word "Queen" in "The rock band Queen" and "Queen Victoria" 
would both be assigned the same 300-dimensional vector).  The fact that word2vec is basically just a lookup table 
from words to vectors is the reason for this limitation.
{% endcapture %}
{% include problem.html problem=problem solution=solution %}

## BERT architecture

Now we're going to learn about an approach to encoding words as vectors called BERT (Bidirectional Encoder 
Representations from Transformers).  BERT is one of the most important papers in all of the natural language 
processing literature (cited over 150,000 times!).  As such, there are many different guides to read to learn about 
it.  We'll suggest a pathway to learn about it in the assignment block to follow (including some questions to ponder)
, but you should feel free to find your own resources and let us know if you find anything particularly good.

{% capture content %}
The original paper that describes BERT, [BERT: Pre-training of Deep Bidirectional Transformers for Language 
Understanding](https://arxiv.org/abs/1810.04805), is quite dense and may not be the easiest place to start.  On the other hand, if you want the most 
detail possible, this is the place to go.  Here are some resources that we recommend diving into.
* [Towards Data Science: A Complete Guide to BERT](https://towardsdatascience.com/a-complete-guide-to-bert-with-code-9f87602e4a11/) (in particular the section 1 and 2).
* The resource above appeared in an earlier form that is useful to us.  Check out [BERT word embeddings tutorial]
  (https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/).  Most of the content is redundant to the 
  first resource, but section 3 (Extracting Embeddings), has some useful stuff.
* [TrAVis: Transformer Attention Visualiser](https://ayaka14732.github.io/TrAVis/?utm_source=chatgpt.com) will allow 
  you to explore visualizations of attention at different levels of a BERT model.  You can explore the default 
  sentence or, if you wait long enough for the model weights to load, you can input your own sentence.
{% endcapture %}
{% include external_resources.html content=content %}


{% capture problem %}
As you are learning about BERT, here are some guiding questions.
* What is BERT's architecture?  Can you draw a diagram showing how a piece of text is processed by BERT?  How do the 
  transformers that we learned about in the last few assignments appear?
* How does the use of transformers allow BERT to overcome the key limitations of the continuous bag of words model 
  (CBOW) discussed earlier in this assignment?
* What does it mean that BERT is a bidirectional architecture?  What does it mean that it is an encoder-only 
  architecture?  What does it mean that GPT is a decoder-only architecture?
* Why are the representations learned by BERT useful in many different language tasks even though the model was 
  trained only on masked language model and next sentence prediction?  What are these two tasks?
* In order to get a sentence-level vector from BERT, what are some potential choices?
* In BERT Word Embeddings Tutorial (linked above), what is the point of section 3.4 (Confirming Contextually 
  Dependent vectrs).  Why does the author conclude that the model was working well based on the experiment that was run?
{% endcapture %}
{% capture solution %}
* The BERT architecture tokenizes the input text into word parts and then adds together a token embedding, a 
  positional embedding, and a segment embedding (whether the word is in the first or second sentence when training 
  on multiple sentences).  The embeddings go through a series of transformer layers before the outputs are used for 
  either masked language modeling or next sentence prediction.
* By using the transformer architecture, BERT is able to adjust the embeddings of tokens based on surrounding words 
  so that the meaning of the word *in context* is more accurately portrayed.  This significantly improves upon CBOW, 
  which always uses the same embedding for a particular word (regardless of context).
* BERT is bidirectional since tokens can attend to tokens that appear to the left or right of them in a sequence.  
  This means that BERT is not very useful for text generation since the training process does not have to extend 
  existing pieces of text by adding new words, but instead fills in missing words.  The lack of suitability for 
  encoding means that BERT is an encoder only architecture (it can encode existing text into vectors but can't 
  decode the vectors to extend an existing piece of text).  GPT is a decoder only since it takes creates a 
  representation of the input text that is useful for decoding the next word.
* The representations are useful for many tasks since masked language modeling and next sentence prediction require 
  a very general and nuanced model of the structure of language. For next sentence prediction, the token 
  representation for a special token called [cls] is used to determine whether two sentences actually appear after 
  each other in the training data or whether they are from different texts.  In masked language modeling task, the 
  system uses the output representation for any tokens that were replaced by the special [mask] token and attempts 
  to predict the actual token using a linear layer followed by softmax.
* For sentence-level representations one can simply average the token-level representations at the output of the 
  last transformer layer.  Another choice is to take the last four transformer outputs, concatenate them, and then 
  average the resultant vectors over each token.
* In this section, the author shows that the encoding of "bank" is most similar between the first two ussages of 
  bank in the following sentence than the last usage and either of the first two "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
  money from the bank vault".  Cosine similarity is used to demonstrate this outcome.
{% endcapture %}
{% include problem.html problem=problem solution=solution %}

# Hands-on with BERT

{% capture problem %}
Work through the [assignment 15 Colab notebook](https://colab.research.google.com/drive/14kQ7J4TaOBko6ewr8N3dyv6g91-giTGj?usp=sharing).
{% endcapture %}
{% capture solution %}
The solution is in the notebook itself.
{% endcapture %}
{% include problem.html problem=problem solution=solution %}