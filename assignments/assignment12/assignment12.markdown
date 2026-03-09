---
title: "Assignment 12: Generative Pre-Trained Transformers (GPTs) Part 1"
toc_sticky: true 
toc_h_max: 1
layout: problemset
due_on_class: 14
published: true
no_solutions: true
---

# Learning Objectives

{% capture content %}
* Learn about the concept of self-attention in neural networks and the role it plays in Generative Pre-trained Transformers (GPTs)
* Implement self-attention in Pytorch
{% endcapture %}
{% include learning_objectives.html content=content %}

{% capture problem %}
There are no formal exercises this time.  If you are in assessment B, please turn in your notes or other 
investigations related to the three videos below.

(totally optional), if you want some practice in the mechanics of self-attention, you might consider the exercise 1 
and 2 from [the 2024 version of this assignment](https://olinml2024.github.io/assignments/assignment12/assignment12). 
{% endcapture %}
{% include problem.html problem=problem %}

# Demystifying GPT

This assignment and the next one are building towards the goal of demystifying large language models (LLMs) like ChatGPT.  While we won't be able to learn everything there is to know about these models, we will be learning, in-depth, about the concept of Generative Pre-Trained Transformers (GPTs).  We hope that by seeing the GPT mechanism up close, you are able to develop a better understanding of how LLMs work, giving you the option to explore LLMs further in your final projects.  You'll also learn some useful, generalizable tricks for text processing along the way.

The roadmap for our work (over this and the next assignment) is that we are going to use two video resources.  First, we'll watch a sequence of two videos from 3B1B that will help us build a conceptual understanding of GPTs through a visual approach. The second, is a walkthrough of how to turn our conceptual understanding into an implementation of a GPT in Pytorch (we'll use NanoGPT from Andrej Karpathy for that).

# Word Embeddings and Predicting the Next Word

{% capture externalresource %}
Let's start off by watching the 3B1B video [How large language models work, a visual intro to transformers](https://www.youtube.com/watch?v=wjZofJX0v4M).

Here are some of the key things we would like you to take away from this video.
* That text can be tokenized in different ways (either as letters, chunks of words, or whole words)
* How predicting the next token (or word) given a piece of text can be used repeatedly to do text completion.
* That we can use the concept of embeddings to represent tokens in a high-dimensional space (make sure you understand how this connects to word embeddings)
* Why the context that surrounds a word might be important for updating its embedding vector (e.g., to disambiguate between multiple meanings of the same word).
* That the last layer of a GPT model maps from the embedding space to a real number for each possible next token (this is called the "unembedding matrix" in the video).  These numbers are called "logits".
* To take our real numbers from the previous step into a probability of the next token, we use the softmax function.
* Make a note of what materials are review from this video (based on things we've already done).
{% endcapture %}
{% include external_resources.html content=externalresource %}

# Self-attention Under the Hood

Hopefully, you found that video to connect some dots from the last assignment and set the stage nicely for where we are going next.  Our next move is going to be to watch the next chapter in the 3B1B series on deep learning.  This is where we will meet the concept of self-attention, which is going to be at the heart of our GPT model.

{% capture externalresource %}
Now, let's watch the 3B1B video [Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc).

Here are some of the key things we would like you to take away from this video.
* That the initial embedding of a token also encodes its position (not just the token's identity)
* That it is useful for words to be able to ask questions (query) of other words.
* That queries can be specified as vectors and the answers to those queries can also be specified as vectors (called keys).
* That the degree to which a key answers a query can be determined by taking the dot product of the key vector and the query vector and that we can compute the dot product of each query token and each query key as $QK^\top$.  Note that the way Grant Sanderson (the creator of the video) has defined the matrices Q and K, the correct equation for him woudl be $K^\top Q$ (he discusses this issue in the video's top comment).  In our presentation, we are sticking with the original equation $QK^\top$.
* At 9:04, Grant Sanderson talks about the key attending to the query.  This is backwards from our understanding of how this language is typically used (there is some discussion of this in the comments).  We think of the query as attending to the key.
* Applying a softmax to the matrix of dot products of queries and keys gives us a probability distribution of which tokens each token should attend to.
* That the idea of causal attention (where we are predicting future tokens from past tokens) requires that future tokens are not allowed to send information to past tokens.  Further, to accomplish this goal, we can force entries in our query-key matrix corresponding to future tokens influencing past tokens to negative infinity (before applying softmax).  This is called "masking".
* That the token embeddings are updated by adding the value vectors from other tokens (weighted by attention).  (Note: this is presented in the video through the example of using adjectives to update the meaning of a noun.)
* Note: there is a discussion of how to cut down the number of parameters in the value map by decomposing it into the product of the value up and the value down matrices ($V_{\uparrow}$ and $V_{\downarrow}$).  While this is interesting, and we are happy to talk about it,  we don't advise getting hung up on this detail (we will not be using this architecture in the implementation to follow).  Similarly, don't worry about the note about how the $V_{\uparrow}$ matrices are all combined into one matrix called the output matrix.
* That multiple heads of attention can be used to capture multiple ways in which token embeddings can influence each other.  Note: you shouldn't have a super precise notion of what this means, but you should have a notion that multiple heads of attention might be useful.
{% endcapture %}
{% include external_resources.html content=externalresource %}

# Implementing Self-Attention

{% capture nanogpt %}
Next, we're going to see how we can do that by implementing self-attention in Pytorch.  We are going to consult our old friend Karpathy (of micrograd fame) and go through his video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).  In this assignment, we're going to go from the beginning to time stamp 1:11:39.  Watching videos like this is way more valuable when you actively try things out as the video is unfolding.  To help scaffold this, below we have a sequence of time stamps in the video along with things to think about or try.

Before you start the video, you should probably pull up the [gpt-dev.ipynb Colab notebook](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing) (linked in the video description).

* [10:13](https://youtu.be/kCc8FmEb1nY?t=613): make sure you understand the encoder and decoder for characters.  Try it out in the notebook on some sequences you feed in.
* [13:45](https://youtu.be/kCc8FmEb1nY?t=834): (something to ponder if you'd like, but not super critical) think through what Karpathy is doing when choosing the train / test split.  Is it a good idea to choose the first 90% of the data as train and the last 10% as test?
* [16:59](https://youtu.be/kCc8FmEb1nY?t=1019): make sure to understand the role of ``block_size`` as an upperbound on context length as well as the importance of extracting shorter contexts for training to allow the transformer to generate text starting from just a little bit of context.
* [20:59](https://youtu.be/kCc8FmEb1nY?t=1259): make sure to notice how the print outs of "inputs" and "targets" relate to each other.  Notice that targets(i,j) is what needs to be predicted given the context represented in the $i$th row of inputs up to column $j$.
* [22:44](https://youtu.be/kCc8FmEb1nY?t=1364): at this point Karpathy introduces the bigram language model.  The implementation of this Bigram model is Karpathy's way of starting with a simple model and gradually transforming it into a GPT.  This move may be a little bit unintuitive given where we are coming from, but we think it will all gel as the video goes on.  While we haven't seen the bigram model in this class, it's a pretty straightforward idea.  Imagine training a multiclass logistic regression model (linear layer followed by a softmax) that predicts the probability of the next token given the current token.  To represent these probabilities, we'll use a lookup table (implemented as a pytorch embedding) where the entry $i$, $j$ will be larger (more positive) if token $j$ often follows token $i$ and negative if token $j$ is unlikely to follow token $i$.  The entries of this lookup table will be learned from data (these would be the weights in our logistic regression model).
* [27:25](https://youtu.be/kCc8FmEb1nY?t=1645): you may want to play around (meaning running code in the notebook) with the the ``tensor.view`` function to get a sense for how Karpathy is using it to "unroll" the tensor with dimensions ``B, T, C``.
* [28:13](https://youtu.be/kCc8FmEb1nY?t=1693): notice that Karpathy is actually passing the loss as an output from the forward function.  That's a bit different to what we've been doing, but it's just a stylistic difference.  Don't get to hung up on it.
* [29:07](https://youtu.be/kCc8FmEb1nY?t=1747): Karpathy shows code for generating text (basically, continuously feeding the models predictions back into itself).  How this happens is a bit beside the point for us, so we recommend not worrying about the details of how he does this.
* [35:34](https://youtu.be/kCc8FmEb1nY?t=2134): now we are setting up our training loop.  This should look very familiar to what we've done earlier in this class.
* [40:17](https://youtu.be/kCc8FmEb1nY?t=2417): we've now transitioned to using a script.  We are estimating loss by averaging over multiple batches.  This is to avoid computing loss on the entire training set (which we've tended to do since our datasets have been relatively small).  Notice the cool decorator he uses on the ``estimate_loss`` function though (that could be handy to avoid having to using ``with torch.no_grad():``)
* [43:16](https://youtu.be/kCc8FmEb1nY?t=2596): notice that Karpathy is now switching to thinking of embedding the tokens in a space (in this case a 2-dimensional space) rather than using the embeddings as a convenient way to implement a bigram model.  This is similar to what we did when we thought about embeddings is the last assignment.  Instead of computing embeddings using ``nn.Embedding``, we're just generating them randomly to allow us to focus on the machinery of self-attention.
* [45:12](https://youtu.be/kCc8FmEb1nY?t=2712): our old friend the bag of words!  As mentioned in the video, we're only doing this simple averaging step as a brief stepping stone to the attention mechanism we learned about in the 3B1B videos.
* [47:48](https://youtu.be/kCc8FmEb1nY?t=2868): Karpathy really breaks this down nicely.  We recommend you interact with this toy example by running it yourself and making sure you understand the connection between the code and the matrix math.
* [53:35](https://youtu.be/kCc8FmEb1nY?t=3215): a quick note that if you actually run this code ``torch.allclose`` will actually give false!  Presumably some of the default values have changed in pytorch since this video was made.  Passing the keyword argument ``atol=10**-7`` along with the two matrices should give you ``True``.
* [55:35](https://youtu.be/kCc8FmEb1nY?t=3332): this should look familiar!  This is the masking we saw earlier.
* [59:10](https://youtu.be/kCc8FmEb1nY?t=3550): now we are making our bigram model look more like self-attention!  Notice how we are introducing the idea of ``n_embd`` to capture the number of embedding dimensions (this was 2 in the toy problem we did earlier).
* [1:00:17](https://youtu.be/kCc8FmEb1nY?t=3617): the version run at this point is still not doing any attention, but we have added some of the machinery necessary to implement self-attention.
* [1:00:57](https://youtu.be/kCc8FmEb1nY?t=3657): we are introducing position embedding, which was mentioned briefly in 3B1B videos since it can be important to self-attention.
* [1:05:11](https://youtu.be/kCc8FmEb1nY?t=3911): we now introduce the variable ``head_size``, which we previously referred to as the query dimension ($n_q$).  Also, not that if we didn't set ``bias=False`` we would have a constant added to the computation of our queries and keys, which we don't want.
* [1:07:08](https://youtu.be/kCc8FmEb1nY?t=4028): the way that the multiplication of two tensors works is a bit confusing for us, but hopefully we can leverage what we know about matrix multiplication.  If you want to go into this, you can check out [Understanding Broadcasting in Pytorch](https://www.geeksforgeeks.org/understanding-broadcasting-in-pytorch/).
* [1:11:37](https://youtu.be/kCc8FmEb1nY?t=4297): we made it to our stopping point for this assignment.  Look at the code to compute ``out``.  Can you connect the dots to the equation we learned about earlier for computing the output of our attention head, $softmax(mask(\mathbf{Q}\mathbf{K}^\top))\mathbf{V}$, and see how it corresponds?
{% endcapture %}
{% include external_resources.html content=nanogpt %}