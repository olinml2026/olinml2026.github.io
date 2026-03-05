---
title: "Day 12: Project Shareout and Starting Text as Data"
toc_sticky: true 
toc_h_max: 1
layout: problemset
published: true
---

{% capture agenda %}
* 3:45-4:05pm: Share out about small data projects! 
* 4:05-4:15pm: Key takeaways from the project
* 4:15-4:35pm: Key concepts in Learning as Optimization
* 4:35-5:25pm: Shifting into text as data
{% endcapture %}
{% include agenda.html content=agenda %}

# Share out about small data projects! 

We want to celebrate and share about your small data mini-projects. Hooray, you did it! We'll do a little share-out so that you can learn from each other and give high fives!


## Share at tables

Let's start out by sharing what you did with the folks around you.  Take about 2 minutes each to go over what you did for the project and how it turned out.

## If you feel inclined, step up to the front

If you have something you want to show the class (could be something you are proud of, a hard fought lesson, advice, etc.), please jump up and connect to the projector.

# Key takeaways from the mini-project

Let's mix up the seating.  Please sit with people you don't normally sit with.  As a group make a list of key takeaways for this project.  Some good prompts would be:

1. New things I learned through this project are...
2. I solidified my knowledge of...
3. Next time I use machine learning I will be sure to...

Take about 5 minutes for this in groups and then we'll do a 5 minute share out.

# Key concepts in Learning as Optimization

Before we turn to the next unit of the course, let's take stock of what we've learned in this past module 
(Learning as Optimization).  We'd like you to create a concept map of the main ideas in the learning as optimization 
module.  From this concept map, please make a list of what you think the key takeaways are from this module.  For 
your reference, we've made our own list (see this [Learning As Optimization Takeaways](../assignments/assignment10/LearningAsOptimizationTakeaways) for details about what we expect). 

# Important Problems in the Field of Text Processing

Before we get into how to process text, let's ask *why* we might want to process text.  Perhaps this seems like a
silly question given the fact that everywhere you turn these days folk are talking about processing text with large  
language models (LLMs).  We're going to go over a few of the specific problems that arise in a field called Natural
Language Processing (or NLP for short).  NLP is a field concerned with, not surprisingly, processing and making 
sense of natural language.  Don't let the term "natural language" confuse you, all we mean here is that we want to 
be able to process text that is written in natural form (i.e., how humans communicate).  In this case the world 
"natural" might be seen as a contrast to the notion of processing text that is constructed in some specific way as 
to be easily interpretable by a computer (e.g., a programming language is a good example).

Here are some examples (not even close to an exhaustive list) of NLP problems that are commonly studied in the field.

* **Machine translation:** translating text from one language to another.
* **Text completion:** given the beginning of a piece of text, complete it (this is at the heart of LLMs)
* **Question answering:** given a question, answer it in natural language (again this is a big part of LLMs)
* **Named entity recognition:** "seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc." ([source](https://en.wikipedia.org/wiki/Named-entity_recognition))
* **Sentence parsing:** given a sentence, determine parts of speech and how they relate to each other
* **Sentiment analysis:** given a sentence, determine whether the sentiment contained is positive or negative (this could be generalized to emotion detection or transferred to thinking about other types of text classification, e.g., spam filters for email).


## Text Processing Beyond Natural Language

Many of the same techniques we will be learning about can be used to process text data other than natural language.  
Examples of this sort of text data could be genomic sequences (where each symbol in the sequence consists of 
nucleotides A, C, T, and G), amino acid chains (where each symbol is one of the 20 amino acids present in the human 
body), structured text (e.g., Python code), etc.  For example, the Google's DeepMind team's [AlphaFold program for protein structure prediction just led to a Nobel prize in chemistry](https://www.nature.com/articles/d41586-024-03214-7).  [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) predicts protein structure from an Amino acid chain.  We won't be going into this sort of text processing in this module (although some of the methods we will learn could be adapted fairly easily).  If you are interested in the idea of processing non-linguistic text, this might be a fruitful topic for a final project.


# Text as Data

The theme of this module is text as data.  In this module we will begin to explore how we can use machine learning
approaches to process text in order to solve problems (e.g., text classification or language translation).  
Throughout this module, we will learn different methods to convert text to numbers that can be operated upon using
the machine learning techniques we learned about in the last module (e.g., logistic regression and MLPs).

## Key Properties

Before we dive into some of the key applications of machine learning for text processing, let's take some time to
think about what makes processing text different than much of the data we've looked at thus far.

### Text consists of symbols

Pieces of text are comprised of symbols.  For example, the text you are reading right now consists of symbols that
include letters, numbers, punctuation, and other special characters.  Perhaps the most important distinction for us
as machine learning practitioners is that these symbols do not necessarily have a meaningful numerical
representation that we can use for learning.  As we move forward in this module, we're going to learn different
methods for changing these symbols into useful numerical representations so that we can use techniques like logistic
regression and MLPs for further processing.  It's also worth mentioning that when representing text we can also
choose the symbols that we use.  Some models treat each letter as an individual symbol, and others treat each word
as a symbol.  Other models treat parts of words as symbols.  We'll be digging into all of this in a few assignments.

### Text has sequential structure

When we first met the supervised learning problem, we represented our input to the model as a d-dimensional vector
$\mathbf{x}$.  Each of the dimensions of this vector represented some characteristic of the data.  In the logistic
regression model and the MLP, each dimension of $\mathbf{x}$ was treated more-or-less independently.  We did not
assume any specific relationship between $x_i$ and $x_j$ (we could just as easily have shuffled the dimensions of
the data and our learning approaches wouldn't have behaved any differently).  When processing text, we need to
consider that pieces of text have sequential structure.  The order of the symbols matters.  Our first attempts (in
this assignment) to map machine learning onto text processing will not do a great job encoding this sequential
structure, but as we move through the module we will begin to represent this sequential structure in important ways.

### Text has variable length

Again, thinking back to our input vector $\mathbf{x}$, it had a fixed number of dimensions (we used $d$ to refer to
the number of dimensions).  Pieces of text consist of sequences of symbols *of variable length*.  As a concrete
example, later in this document you'll learn about sentiment analysis (predicting if a piece of text is positive or
negative) from text.  The individual pieces of text will contain varying numbers of symbols.  Our machine learning
methods must handle this variability, and so far it's not obvious how we can make this happen (but we'll see one way
by the end of this assignment).

# Vectorizing Text Two Ways

## Bag of Words

Next, we're going to learn about our first technique for adapting the machine learning approaches from the previous 
module to processing text.  In doing so, we're going to find ways of dealing with some of the unique features of 
text that initially might seem to make text incompatible with the techniques we've learned about.  Our first 
technique of the module is called "bag of words," and it deals with two important challenges we've already discussed 
in using machine learning methods with text.  First, it converts the symbols in a piece of text into a numerical 
representation.  Second, the technique is able to handle pieces of text that are variable in length.  We hope you 
will enjoy these great external resources for learning about bag of words.

{% capture resources %}
Let's learn about bag of words!  Begin by watching
[a video from IBM called "What is Bag of Words?"](https://www.youtube.com/embed/pF9wCgUbRtc?si=zd1AYDQTJifqLtcZ).  
Towards the end, this video gets into two more advanced topics that we'll be digging into shortly.  The first is
tf-idf and you'll learn about that in the notebook.  The second is the idea of word embeddings (or word2vec), and
you'll see that in the assignment after this one.  We point this out since we want you to focus on the bag of words
content and avoid getting thrown off by this other content.  If you want one more (shorter video), we also recommend
[this video from Socratica](https://www.youtube.com/embed/kLMhePA3BiY?si=MEfYE_SyhzkGBnch).
{% endcapture %}
{% include external_resources.html content=resources %}

{% capture problem %}
As a quick check of your understanding, encode the following three pieces of text using bag of words.  What would
you need to do to normalize the data?  What does it mean that the bag of words is a sparse representation?  How do
you see that in your solution to the exercise?

1. goodnight moon
2. goodnight cow jumping over the moon
3. and a little toy house and a young mouse
4. and goodnight mouse
   {% endcapture %}
   {% capture solution %}

If we examine the texts as a whole, we can identify the unique words that occur and assign each word to a dimension
in our bag of words vector.  As long as we're consistent in how we do so, It doesn't matter how we assign words to
vector dimensions (we could shuffle the rows of the table below, and we would still have a valid bag of words
representation).  Here is what the sentences could look like in bag of words form.

| dimension | word    | text 1 | text 2 | text 3 | text 4 |
| -------- | ------- | ------- | ----- | ----- | ---- |
| 1 | goodnight  |  1   | 1 | 0 | 1 |
| 2 | moon |   1 | 1 | 0 | 0   |
| 3 | cow    |  0 | 1 | 0 | 0   |
| 4 | jumping  | 0 | 1  |  0 | 0   |
| 5 | over    |  0 | 1 | 0 | 0   |
| 6 | the    |  0 | 1 | 0 | 0   |
| 7 | and    |  0 | 0 | 2 | 1   |
| 8 | a    |  0 | 0 | 2 | 0   |
| 9 | little    |   0 | 0 | 1 | 0  |
| 10 | toy    |  0 | 0 | 1 | 0   |
| 11 | house    |  0 | 0 | 1 | 0   |
| 12 | young    |  0 | 0 | 1 | 0   |
| 13 | mouse    |  0 | 0 | 1 | 1  |


If we were to normalize these representations, we would divide each column by the sum of the column (i.e., the total
number of words in each piece of text).

The bag of words representation is sparse as most of the entries in the table are 0.  If we had a larger vocabulary
the sparsity would be even more apparent (a higher proportion of entries that are 0).

{% endcapture %}

{% include problem.html problem=problem solution=solution %}

# Motivating Example for Our Next Assignment

We'll be going over the concept of word embeddings using a combination of a mini-lecture and [a Colab notebook](https://colab.research.google.com/drive/1a-P0IEZs-wV4VZpdznaXpGJU_qJipqL_?usp=sharing).
