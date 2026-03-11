---
title: "Day 14: Self-Attention in Transformers"
toc_sticky: true 
toc_h_max: 1
layout: problemset
published: true
---

{% capture agenda %}
* 3:45-3:55pm: Assignment debrief
* 3:55-4:15pm: Walkthrough of attention by hand problems
* 4:15-4:35pm: Overview of first-half of NanoGPT
* 4:35-5:25pm: Start on next assignment(s)
{% endcapture %}
{% include agenda.html content=agenda %}

# Assignment Debrief

{% capture problem %}
With people around you, given an input sentence, describe each of the computations performed to arrive at the 
self-attention matrix (i.e., how much token i attends to token j).
{% endcapture %}
{% include problem.html problem=problem %}

# Walkthrough of attention by hand problems


{% capture problem %}
Let's use a toy problem to make sure we have a handle on the mechanics of self-attention.  Instead of words, let's think of individual letters as our tokens (again, sorry for this sleight-of-hand.  We are doing this to make the problem as simple as possible to highlight the important bits of self-attention.  We'll also be using a resource called NanoGPT that will implement a GPT, at first, on the character level).  Let's imagine that we want our attention head to take in a sequence of letters and compute for each token whether a consonant has occurred at any point up to and including the current token.  Here are some examples.

1. Input text: "eaeia", our attention head should output no, no, no, no, no (none of our token have the property that they are or are preceded by a consonant).
2. Input text: "ccrs", our attention head should output yes, yes, yes, yes (all tokens either are or are preceded by a consonant)
3. Input text: "aeri", our attention head should output no, no, yes, yes (starting with the third token, "r", we have at least one consonant).

We haven't quite defined how the responses "no" and "yes" will be represented as vectors, but we will get to that shortly.

Let's use a tokenization scheme where each letter is mapped to its position in the alphabet (starting with $a \rightarrow 0$ and ending with $z \rightarrow 25$).

{% capture parta_prob %}
Explain what each of the features (the rows) of the input tokens (the columns) in the embedding matrix $\mathbf{W_E}$ 
captures.

\$\$
\mathbf{W_E} = \begin{bmatrix} 1 & 0 & 0 &  0 & 1 & 0 &  0 &  0 & 1 &  0 &  0 &  0 &  0 &  0 & 1 &  0 &  0 &  0 &  0 &  0 
& 1 & 0 &  0 &  0 &  0 &  0 \\\\ 0 &  1& 1 &  1 & 0 & 1 &  1 &  1 & 0 &  1 &  1 &  1 &  1 &  1 & 0 &  1 &  1 &  1 &  1 &  1 & 0 & 1 &  1 &  1 &  1 &  1  \end{bmatrix}
\$\$
{% endcapture %}
{% capture parta_sol %}
The first row of the matrix encodes whether the token is a vowel (1) or consonant (0).  The second row of the matrix encodes whether the token is a consonant (1) or a vowel (0).
{% endcapture %}
{% include problem_part.html subpart=parta_prob solution=parta_sol label="A" %}

{% capture partb_prob %}
Define a query ($\mathbf{W_q}$) and key ($\mathbf{W_k}$) matrix pair that causes all letters to attend to consonants.

$\mathbf{W_q}$ and $\mathbf{W_k}$ are both matrices with $n_{q}$ rows and $n_{e}$ columns, where $n_q$ is the query 
dimension (you can choose this) and $n_e$ is the dimensionality our embeddings (in this example, 2).

Hint 1: You should be able to solve the problem with $n_{q} = 1$ (that is, the key and query matrices are both 1 row and 2 columns).

Hint 2: The key equation you'll want to use is that the degree to which token $i$ attends to token $j$ can be computed from the embeddings $\mathbf{r}_i$ and $\mathbf{r}_j$ (these would be found in the appropriate column of $\mathbf{W_E}$) of tokens $i$ and $j$ respectively using the following formula.

\$\$
\begin{aligned}
attention &= (\mathbf{W_q} \mathbf{r}_i) \cdot (\mathbf{W_k} \mathbf{r}_j)
\end{aligned}
\$\$
{% endcapture %}
{% capture partb_sol %}
Let's define the matrices as follows.

$$
\begin{aligned}
\mathbf{W_q} &= \begin{bmatrix} 1 & 1 \end{bmatrix} \\
\mathbf{W_k} &= \begin{bmatrix} 0 & 5 \end{bmatrix}
\end{aligned}
$$

Notice how no matter whether we have a consonant or a vowel, our query will always be $1$.  This makes sense since all tokens issue the same query (is there a consonant in front of me).  In contrast, our keys will only be non-zero if the token is a consonant.  This is also consistent with what we want.

Taking it for a test spin, let's look at the different cases.

* query is vowel and key is vowel $$\bigg (\mathbf{W_q}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \cdot \bigg(\mathbf{W_k} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg) = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \cdot \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg) = (1)(0) = 0$$
* query is consonant and key is vowel $$\bigg (\mathbf{W_q}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \cdot \bigg(\mathbf{W_k} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg) = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \cdot \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix}\bigg) = (1)(0) = 0$$
* query is vowel and key is consonant $$\bigg (\mathbf{W_q}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \cdot \bigg(\mathbf{W_k} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg) = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} \bigg ) \cdot \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg) = (1)(5) = 5$$
* query is consonant and key is consonant $$\bigg (\mathbf{W_q}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \cdot \bigg(\mathbf{W_k} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg) = \bigg (\begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 0 \\ 1 \end{bmatrix} \bigg ) \cdot \bigg(\begin{bmatrix} 0 & 5 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\bigg) = (1)(5) = 5$$

Why $5$?  This helps make the attention to consonants higher relative to attention to vowels (remember, this has to get passed through a softmax).


{% endcapture %}
{% include problem_part.html subpart=partb_prob solution=partb_sol label="B" %}

{% capture partc_prob %}
Come up with a short sequence of characters, $s$, consisting of some vowels and some consonants (keep the length pretty small).  Compute the matrix of all queries corresponding to your sequence, $\mathbf{Q}$, where the number of rows of $\mathbf{Q}$ is equal to the number of tokens (the length of $s$) and the number of columns is equal to the query dimension.  Compute the matrix of all keys corresponding to your sequence, $\mathbf{K}$, where the number of rows of $\mathbf{K}$ is equal to the number of tokens (the length of $s$) and the number of columns is equal to the query dimension.  Compute the (pre-masking) attention of each token to each other token using the formula $\mathbf{Q} \mathbf{K}^\top$.  Apply masking to ensure that keys (columns) corresponding to later tokens do not influence earlier queries (rows).  Note: that the visualization in the 3B1B video (at [this time stamp](https://youtu.be/eMlx5fFNoYc?t=514)) has this matrix laid out with query tokens as columns and the keys as rows (we wanted to let you know to minimize confusion).  Apply a softmax across each row (as before, this is shown on columns in the 3B1B video) to determine a weight for each token and show the resultant matrix.
{% endcapture %}

{% capture partc_sol %}
Let's take our string to be $s = \text{abcce}$.

Step 1: Compute our embeddings by picking out appropriate columns of our matrix. $r_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$, $r_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$, $r_3 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$, $r_4 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$, and $r_5 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

Step 2: Compute each query using the formula $\mathbf{W_q} \mathbf{r}_i$ and each key using the formula $\mathbf{W_k} \mathbf{r}_i$ and put each query as a row to form $\mathbf{Q}$ and each key as a row to form $\mathbf{K}$.

$$
\begin{aligned}
\mathbf{Q} &= \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \\ 1 \end{bmatrix} \\
\mathbf{K} &= \begin{bmatrix} 0 \\ 5 \\ 5 \\ 5 \\ 0 \end{bmatrix}
\end{aligned}
$$

Step 3: Compute the unmasked attention $\mathbf{Q} \mathbf{K}^\top$.

$$
\begin{aligned}
\mathbf{Q} \mathbf{K}^\top &= \begin{bmatrix} 0 & 5 & 5 & 5 & 0 \\ 0 & 5 & 5 & 5 & 0 \\ 0 & 5 & 5 & 5 & 0 \\ 0 & 5 & 5 & 5 & 0 \\ 0 & 5 & 5 & 5 & 0 \end{bmatrix}
\end{aligned}
$$

Step 4: Mask the matrix so that future tokens can't influence past tokens.

$$
\begin{aligned}
mask(\mathbf{Q} \mathbf{K}^\top) &= \begin{bmatrix} 0 & -\infty & -\infty & -\infty & -\infty \\ 0 & 5 & -\infty & -\infty & -\infty \\ 0 & 5 & 5 & -\infty & -\infty \\ 0 & 5 & 5 & 5 & -\infty \\ 0 & 5 & 5 & 5 & 0 \end{bmatrix}
\end{aligned}
$$

Step 5: Take softmax along the rows.

$$
\begin{aligned}
softmax(mask(\mathbf{Q} \mathbf{K}^\top)) &= \begin{bmatrix}    1 &  0 &  0 & 0 & 0 \\ 0.0067 &  0.9933  & 0   &      0   &    0 \\ 0.0034   & 0.4983 &   0.4983     &    0     &    0 \\   0.0022  &  0.3326  &  0.3326  &  0.3326    &     0 \\ 0.0022  &  0.3318  &  0.3318  &  0.3318  &  0.0022 \end{bmatrix}
\end{aligned}
$$


{% endcapture %}
{% include problem_part.html subpart=partc_prob solution=partc_sol label="C" %}

{% capture partd_prob %}
Define the value for the $i$th token as $\mathbf{W_V} \mathbf{r}_i$ where $\mathbf{W_V}$ is the identity matrix and $\mathbf{r}_i$ is the embedding of the token.  Construct the matrix $\mathbf{V}$ by computing the values of each token using the formula $\mathbf{W_V} \mathbf{r}_i$ and then transforming each value to a row of a matrix.  Show that taking your attention matrix from Part C and multiplying it on the right by $\mathbf{V}$ computes the output of the attention head which will give a vector close to $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ if no consonants preceded a token and $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$ if at least one consonant preceded a token.
{% endcapture %}

{% capture partd_sol %}
The values are going to be the same as our embeddings.  We can lay them out as the rows of $\mathbf{V}$.

$$
\begin{aligned}
\mathbf{V} &= \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}
\end{aligned}
$$

We get the final outputs of our attention head by multiplying our matrix from part C by $\mathbf{V}$.

$$
\begin{aligned}
\begin{bmatrix}    1 &  0 &  0 & 0 & 0 \\ 0.0067 &  0.9933  & 0    &     0   &    0 \\ 0.0034   & 0.4983 &   0.4983     &    0     &    0 \\   0.0022  &  0.3326  &  0.3326  &  0.3326    &     0 \\ 0.0022  &  0.3318  &  0.3318  &  0.3318  &  0.0022 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix} &= \begin{bmatrix} 1.0000     &    0 \\ 0.0067  &  0.9933 \\ 0.0034  &  0.9966 \\ 0.0022  & 0.9978 \\  0.0045  &  0.9955 \end{bmatrix}
\end{aligned}
$$

{% endcapture %}
{% include problem_part.html subpart=partd_prob solution=partd_sol label="D" %}

{% capture parte_prob %}
Suppose you wanted the attention head to determine the proportion of consonants that precede (rather than just whether a consonant precedes a word or not).  How would you modify $\mathbf{W_Q}$ and $\mathbf{W_K}$ to achieve this result?  You should not need to change $\mathbf{V}$.
{% endcapture %}
{% capture parte_sol %}
We could keep $\mathbf{W_Q} = \begin{bmatrix} 1 & 1 \end{bmatrix}$ the same.  We can now modify the key so that all tokens have the same key (all respond to the query) by setting $\mathbf{W_K} = \begin{bmatrix} 1 & 1 \end{bmatrix}$. Let's turn the crank.

$$
\begin{aligned}
\mathbf{Q} &= \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \\ 1 \end{bmatrix} \\
\mathbf{K} &= \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \\ 1 \end{bmatrix}
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{Q} \mathbf{K}^\top &= \begin{bmatrix} 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \end{bmatrix}
\end{aligned}
$$

$$
\begin{aligned}
mask(\mathbf{Q} \mathbf{K}^\top) &= \begin{bmatrix} 1 & -\infty & -\infty & -\infty & -\infty \\ 1 & 1 & -\infty & -\infty & -\infty \\ 1 & 1 & 1 & -\infty & -\infty \\ 1 & 1 & 1 & 1 & -\infty \\ 1 & 1 & 1 & 1 & 1 \end{bmatrix}
\end{aligned}
$$

$$
\begin{aligned}
softmax(mask(\mathbf{Q} \mathbf{K}^\top)) &= \begin{bmatrix}    1 &  0 &  0 & 0 & 0 \\ 0.5 &  0.5  & 0   &      0   &    0 \\ 0.3333   & 0.3333 &   0.3333     &    0     &    0 \\   0.25  &  0.25  &  0.25  &  0.25    &     0 \\ 0.2  &  0.2  &  0.2  &  0.2  &  0.2 \end{bmatrix}
\end{aligned}
$$

Finally, combine our attention with our values (since they haven't changed from part D, let's just use those).
$$
\begin{aligned}
\begin{bmatrix}    1 &  0 &  0 & 0 & 0 \\ 0.5 &  0.5  & 0   &      0   &    0 \\ 0.3333   & 0.3333 &   0.3333     &    0     &    0 \\   0.25  &  0.25  &  0.25  &  0.25    &     0 \\ 0.2  &  0.2  &  0.2  &  0.2  &  0.2 \end{bmatrix}\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix} &= \begin{bmatrix}  1.0000   &      0 \\    0.5000  &  0.5000 \\ 0.3333  &  0.6667 \\   0.2500  &  0.7500 \\   0.4000 &   0.6000 \end{bmatrix}
\end{aligned}
$$

{% endcapture %}
{% include problem_part.html subpart=parte_prob solution=parte_sol label="E" %}

{% endcapture %}
{% include problem_with_parts.html problem=problem %}

Next, let's see how a position embedding might help us.

{% capture problem %}
Suppose we want our attention head to take in a sequence of letters and output the vector $$\begin{bmatrix} 1 \\ 0 \end{bmatrix}$$ if there is a consonant at position 1 (where 1 is the first position in the sequence) and $$\begin{bmatrix} 0 \\ 0 \end{bmatrix}$$ otherwise.

1. Input text: "eacia", our attention head should output $$\begin{bmatrix} 0 \\ 0 \end{bmatrix}$$, $$\begin{bmatrix} 0 \\ 0 \end{bmatrix}$$, $$\begin{bmatrix} 0 \\ 0 \end{bmatrix}$$, $$\begin{bmatrix} 0 \\ 0 \end{bmatrix}$$, $$\begin{bmatrix} 0 \\ 0 \end{bmatrix}$$ (token 1 is a vowel).
2. Input text: "ccrs", our attention head should output $$\begin{bmatrix} 1 \\ 0 \end{bmatrix}$$, $$\begin{bmatrix} 1 \\ 0 \end{bmatrix}$$, $$\begin{bmatrix} 1 \\ 0 \end{bmatrix}$$, $$\begin{bmatrix} 1 \\ 0 \end{bmatrix}$$ (the first token is a consonant).

Let's use the same tokenization scheme as in the previous exercise. That is, each letter is mapped to its position in the alphabet (starting with $a \rightarrow 0$ and ending with $z \rightarrow 25$).

{% capture parta_prob %}
Explain what each of the features (the rows) of the input tokens (the columns) in the embedding matrix $\mathbf{W_E}$ captures.

<div>
$$
\mathbf{W_E} = \begin{bmatrix} 1 & 0 & 0 &  0 & 1 & 0 &  0 &  0 & 1 &  0 &  0 &  0 &  0 &  0 & 1 &  0 &  0 &  0 &  0 &  0 & 1 & 0 &  0 &  0 &  0 &  0 \\ 0 &  1& 1 &  1 & 0 & 1 &  1 &  1 & 0 &  1 &  1 &  1 &  1 &  1 & 0 &  1 &  1 &  1 &  1 &  1 & 0 & 1 &  1 &  1 &  1 &  1 \\ 0 &  0 & 0 &  0 & 0 & 0 &  0 &  0 & 0 &  0 &  0 &  0 &  0 &  0 & 0 &  0 &  0 &  0 &  0 &  0 & 0 & 0 &  0 &  0 &  0 &  0  \end{bmatrix}
$$
</div>

We can also specify our position embeddings for each token position (we'll stop at position $8$ since the pattern should be obvious).  Explain what the positional embedding matrix is representing.

<div>
$$
\mathbf{W_P} = \begin{bmatrix} 0 & 0 & 0 &  0 & 0 & 0 &  0 &  0 \\ 0 & 0 & 0 &  0 & 0 & 0 &  0 &  0  \\ 1 & 0 & 0 &  0 & 0 & 0 &  0 &  0  \end{bmatrix}
$$
</div>

{% endcapture %}
{% capture parta_sol %}
We have the same embedding as the previous problem but we've added a dimension that is always zero for the token embedding.  The positional embedding places a 1 in this dimension if the position is 1.
{% endcapture %}
{% include problem_part.html subpart=parta_prob solution=parta_sol label="A" %}

{% capture partb_prob %}
Define a query ($\mathbf{W_q}$) and key ($\mathbf{W_k}$) matrix pair that causes all letters to attend to only the first position in the sequence.  In this example, each key might emit the same query (no matter if it is a consonant or value), but the key would only match in the case where the key corresponds to the first token in the sequence.

$\mathbf{W_q}$ and $\mathbf{W_k}$ are both matrices with $n_{q}$ rows and $n_{e}$ columns, where $n_q$ is the query dimension (you can choose this) and $n_e$ is the dimensionality our embeddings (in this example, 3).

Hint 1: You should be able to solve the problem with $n_{q} = 1$ (that is, the key and query matrices are both 1 row and 2 columns).

Hint 2: The key equation you'll want to use is that the degree to which token $i$ attends to token $j$ can be computed from the embeddings (both position and token embedding) $\mathbf{r}_i$ and $\mathbf{r}_j$ (these would be found in the appropriate columns of $\mathbf{W_E}$ and $\mathbf{W_P}$) of tokens $i$ and $j$ respectively using the following formula.

<div>
$$
\begin{aligned}
attention &= (\mathbf{W_q} \mathbf{r}_i ) \cdot (\mathbf{W_k} \mathbf{r}_j)
\end{aligned}
$$
</div>

{% endcapture %}
{% capture partb_sol %}
Let's define the matrices as follows.

<div>$$
\begin{aligned}
\mathbf{W_q} &= \begin{bmatrix} 1 & 1 & 0 \end{bmatrix} \\
\mathbf{W_k} &= \begin{bmatrix} 0 & 0 & 5 \end{bmatrix}
\end{aligned}
$$</div>

Thinking of this intuitively, each token will emit the same query (a value of $1$) no matter if it is a consonant or a vowel.  This is consistent with the fact that all tokens want to attend to the same type of token (the first token).  The key will only be non-zero for tokens that are in the first position (since all others will have a value of $0$ for the final dimension).

We leave it to you to further validate that these matrices will do the job (sorry!).

{% endcapture %}
{% include problem_part.html subpart=partb_prob solution=partb_sol label="B" %}

{% capture partc_prob %}
Come up with a short sequence of characters, $s$, consisting of some vowels and some consonants (keep the length pretty small).  Compute the matrix of all queries corresponding to your sequence, $\mathbf{Q}$, where the number of rows of $\mathbf{Q}$ is equal to the number of tokens (the length of $s$) and the number of columns is equal to the query dimension.  Compute the matrix of all keys corresponding to your sequence, $\mathbf{K}$, where the number of rows of $\mathbf{K}$ is equal to the number of tokens (the length of $s$) and the number of columns is equal to the query dimension.  Compute the (pre-masking) attention of each token to each other token using the formula $\mathbf{Q} \mathbf{K}^\top$.  Apply masking to ensure that keys (columns) corresponding to later tokens do not influence earlier queries (rows).  Note: that the visualization in the 3B1B video (at [this time stamp](https://youtu.be/eMlx5fFNoYc?t=514)) has this matrix laid out with query tokens as columns and the keys as rows (we wanted to let you know to minimize confusion).  Apply a softmax across each row (as before, this is shown on columns in the 3B1B video) to determine a weight for each token and show the resultant matrix.
{% endcapture %}

{% capture partc_sol %}
Let's take our string to be $s = \text{cbcce}$.

Step 1: Compute our embeddings by picking out appropriate columns of our matrices (for both token and position embeddings). $$r_1 = \begin{bmatrix} 0 \\ 1 \\ 1  \end{bmatrix}$$, $$r_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$, $$r_3 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$, $$r_4 = \begin{bmatrix} 0  \\ 1 \\ 0 \end{bmatrix}$$, and $$r_5 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$$.

Step 2: Compute each query using the formula $\mathbf{W_q} \mathbf{r}_i$ and each key using the formula $\mathbf{W_k} \mathbf{r}_i$ and put each query as a row to form $\mathbf{Q}$ and each key as a row to form $\mathbf{K}$.

$$
\begin{aligned}
\mathbf{Q} &= \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \\ 1 \end{bmatrix} \\
\mathbf{K} &= \begin{bmatrix} 5 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}
\end{aligned}
$$

Step 3: Compute the unmasked attention $\mathbf{Q} \mathbf{K}^\top$.

$$
\begin{aligned}
\mathbf{Q} \mathbf{K}^\top &= \begin{bmatrix} 5 & 0 & 0 & 0 & 0 \\ 5 & 0 & 0 & 0 & 0 \\ 5 & 0 & 0 & 0 & 0 \\ 5 & 0 & 0 & 0 & 0 \\ 5 & 0 & 0 & 0 & 0 \end{bmatrix}
\end{aligned}
$$

Step 4: Mask the matrix so that future tokens can't influence past tokens.

$$
\begin{aligned}
mask(\mathbf{Q} \mathbf{K}^\top) &= \begin{bmatrix} 5 & -\infty & -\infty & -\infty & -\infty \\ 5 & 0 & -\infty & -\infty & -\infty \\ 5 & 0 & 0 & -\infty & -\infty \\ 5 & 0 & 0 & 0 & -\infty \\ 5 & 0 & 0 & 0 & 0 \end{bmatrix}
\end{aligned}
$$

Step 5: Take softmax along the rows.

$$
\begin{aligned}
softmax(mask(\mathbf{Q} \mathbf{K}^\top)) &= \begin{bmatrix}    1.0000     &    0     &    0      &   0     &    0 \\   0.9933 &   0.0067    &     0     &    0      &   0 \\   0.9867  &  0.0066  &  0.0066     &    0    &     0 \\    0.9802  &  0.0066  &  0.0066 &   0.0066     &    0 \\  0.9738  &  0.0066  &   0.0066 &   0.0066  &  0.0066 \end{bmatrix}
\end{aligned}
$$


{% endcapture %}
{% include problem_part.html subpart=partc_prob solution=partc_sol label="C" %}

{% capture partd_prob %}
Determine $\mathbf{W_V}$ to compute the value of each token as $\mathbf{W_V} \mathbf{r}_i$.  $\mathbf{V}$ will be formed by laying out each of these values as a row of the matrix. Show that taking your attention matrix from Part C and multiplying it on the right by $\mathbf{V}$ computes the output of the attention head which will give a vector close to $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ if the first token is a consonant and close to $\begin{bmatrix} 0 \\ 0 \end{bmatrix}$ otherwise.

**Hint:** you'll want to construct $\mathbf{V}$ so consonants are mapped to the vector $$\begin{bmatrix} 1 \\ 0 \end{bmatrix}$$ and vowels are mapped to the vector $$\begin{bmatrix} 0 \\ 0 \end{bmatrix}$$.
{% endcapture %}

{% capture partd_sol %}
$$
\begin{aligned}
\mathbf{W_V} &= \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}
\end{aligned}
$$
(Notice how if we have a consonant, regardless of position, our output will be the second column of the matrix.  Similarly, if we have a consonant, the output will be the zero vector).

Applying our formula for the value of each token, $\mathbf{W_V} \mathbf{r}_i$, and transforming these into rows gives us $\mathbf{V}$.

$$
\begin{aligned}
\mathbf{V} &= \begin{bmatrix} 1 & 0 \\ 1 & 0 \\ 1 & 0 \\ 1 & 0 \\ 0 & 0 \end{bmatrix}
\end{aligned}
$$

We get the final outputs of our attention head by multiplying our matrix from part C by $\mathbf{V}$.

$$
\begin{aligned}
\begin{bmatrix}    1.0000     &    0     &    0      &   0     &    0 \\\\   0.9933 &   0.0067    &     0     &    0      &   0 \\\\   0.9867  &  0.0066  &  0.0066     &    0    &     0 \\\\    0.9802  &  0.0066  &  0.0066 &   0.0066     &    0 \\\\  0.9738  &  0.0066  &   0.0066 &   0.0066  &  0.0066 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 0 \\ 1 & 0 \\ 1 & 0 \\ 0 & 0 \end{bmatrix} &= \begin{bmatrix}       1.0000 &  0   \\  1.0000  & 0  \\    1.0000 &   0   \\  1.0000 & 0  \\   0.9934 & 0  \end{bmatrix}
\end{aligned}
$$

{% endcapture %}
{% include problem_part.html subpart=partd_prob solution=partd_sol label="D" %}

{% capture parte_prob %}
Why was it important to have a position embedding in order to get this attention head to behave (i.e., have the output) the way we wanted it to?
{% endcapture %}
{% capture parte_sol %}
Without the position embedding, we wouldn't be able to only attend to the first token.  We could have tried to attend only to consonants, but that would still attend to any consonant (not just ones that are in the first position).
{% endcapture %}
{% include problem_part.html subpart=parte_prob solution=parte_sol label="E" %}

{% endcapture %}
{% include problem_with_parts.html problem=problem %}

# Overview of first-half of NanoGPT

Let's go over [the code checkpoint from the halfway point of the Karpathy video](https://colab.research.google.com/drive/1H5j8YVCuXod8SVr_mCk4C_BX4ukLaDm4?usp=sharing).  I hope it will be helpful to talk 
through some of the main ideas with you all (and answer some questions).

# Upcoming Assignments

* [Assignment 13](../assignments/assignment13/assignment13), which is due tomorrow, involves reading a paper on trust 
  and trustworthiness in machine learning 
  systems.  If you can't read every word of the paper, please at least familiarize yourself with the contents.  
  we'll be discussing some of the key themes in class on Thursday.
* [Assignment 14](../assignments/assignment13/assignment14), involves finishing up the NanoGPT video and the final 
  3B1B.  You'll also learn about the idea of ablation experiments as a way to understand machine learning models.  
  This assignment is due after Spring break.