---
title: "Day 18: Goodbye Text, Hello Images and Convolution"
toc_sticky: true 
toc_h_max: 1
layout: problemset
published: true
---

{% capture agenda %}
* 3:45-4:25pm: Closing out Text as Data 
* 4:25-5:25pm: Starting Images as data
{% endcapture %}
{% include agenda.html content=agenda %}

# Closing out Text as Data Module

After class, I will open up another quality-assessed deliverable on text as data.  As with the model evaluation 
quality-assessed deliverable, it must be done solo and without AI-assistance (you can look at course materials or 
other online resources).  To help you prepare for this assessment, we'd like you to make sure you have a handle on 
the material by working collaboratively with your table to identify any lingering questions.  Below, we have a 
non-exhaustive list of topics for you to think through with your group.  You may consider taking turns going through 
each topic and presenting the main points to each other. 

* What are some key properties of text that one must keep in mind when applying machine learning?
* What is the bag of words approach and why does it have that name?  How do you apply it to a text classification 
  task?  What does it mean for the vectors in bag of words to be sparse?  How does Tf-IDF improve on the basic bag 
  of words model?
* What is word2vec and how is the continuous bag of words model trained?  Give an example of how bias in training 
  data can lead to bias in the vectors learned by word2vec.  What is one method of eliminating this bias?
* How do transformers improve upon models like word2vec?
* Let's return to [the visualization](https://bbycroft.net/llm) of NanoGPT from a few assignments ago.  With folks 
around you, identify aspects of the visualization that you don't understand.  That is, what are the pieces in this diagram that we did not touch on in class thus far?
<details markdown="1">
  <summary>Some of our notes on this visualization</summary>
***Point 1:*** In [this visualization](https://bbycroft.net/llm) we have 3 attention heads.  Each attention head has its own independent matrices for computing keys, queries, and values.

***Point 2:*** The value vectors for each token in each of our 3 attention heads is 16-dimensional.  These 16-dimensional vectors are added together in a weighted fashion (with the weight given by the self-attention matrix) to compute the output of each attention head.

***Point 3:*** We stack the outputs of each attention head to get back to our original 48-dimensional space (the dimensionality of the embedding space is $C=48$).

***Point 4:*** We then take the vector from the previous step and pass it through a projection matrix to translate from whatever representations were learned by each attention head to something that is appropriate to add to the input embedding (via the residual pathway).  Amanda asked a brilliant question about this in class, which was why this translation is needed since all of the attention heads have the same inputs.  This is still a hard question to answer, and Jess Brown did a nice job offering a suggestion that each of the attention heads might learn a different internal meaning of value space (the $V$ matrix), and we need a linear mapping (a matrix) in order to combine these different value spaces (across heads) in a meaningful way. After reviewing the visualization again, there is one more way to explain this.  If we look back at [this section of the 3B1B video](https://youtu.be/eMlx5fFNoYc?t=818) we see two ways to think about computing value vectors in an attention head.

We could (but don't) think of the matrix, $\mathbf{W_V}$, that maps from embeddings to value vectors as a $C \times C$ matrix (where $C$ is the embedding dimension).  As Grant Sanderson, of 3B1B, points out, this approach would use many more parameters to represent the mapping from embeddings to value vectors (versus embeddings to keys or embeddings to queries).  To make the number of parameters similar between these three entities (keys, queries, and values), we can instead think of two steps for computing our value vectors.  First, we use a matrix $\mathbf{V_\downarrow}$ to go from the embedding space to a lower dimensional space (in the visualization we go from $C=48$ to $16$ dimensions).  Second, we use a matrix called $\mathbf{V_\uparrow}$ to go from the 16-dimensional representation back to the $48$ dimensional representation.  As Grant explains, this change to how we compute our value vectors constrains the number of parameters versus having $\mathbf{W_V}$ as a $C \times C$ (48 by 48) matrix.  Mapping this intuition onto our visualization of NanoGPT, we can think of the box labeled ``V Weights`` as playing the role of $\mathbf{V_\downarrow}$ and the box labeled ``Project Weights`` as containing the $\mathbf{V_\uparrow}$ matrices for each of the three attention heads (stacked).

</details>

* When creating a chatbot, why do we use causal self-attention (where tokens can only pay attention to themselves or 
  previous tokens)?  In BERT, why ae we allowed to use bi-directional attention?  What does this say about the types 
  of tasks one might use BERT for versus a GPT model?
* What tasks are used to train the BERT model?  Why would these tasks be useful for learning a general-purpose model 
  of language?
* How does SBERT improve upon BERT?  How do you train an SBERT model?
* Why might finetuning an SBERT model be useful?  How could you evaluate the effects of this finetuning process on 
  performance (either on the new dataset or old datasets)?

# Images as data


{% capture problem %}
Let's discuss:  
What is different about images compared to a set of variables (like in the Titanic data set)? What about compared to text data?
{% endcapture %}
{% include problem.html problem=problem %}

One common application of image data is in medical image processing. Here's a few recent papers, including one about one about clinical trials. 
* [McKinney, S.M., Sieniek, M., Godbole, V. et al. International evaluation of an AI system for breast cancer screening. Nature 577, 89–94 (2020). https://doi.org/10.1038/s41586-019-1799-6](https://www.nature.com/articles/s41586-019-1799-6)
* [Esteva, A., Chou, K., Yeung, S. et al. Deep learning-enabled medical computer vision. npj Digit. Med. 4, 5 (2021). https://doi.org/10.1038/s41746-020-00376-2](https://rdcu.be/dYXIV)
* [Abràmoff, M.D., Lavin, P.T., Birch, M. et al. Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices. npj Digital Med 1, 39 (2018). https://doi.org/10.1038/s41746-018-0040-6](https://www.nature.com/articles/s41746-018-0040-6)

{% capture problem %}
Come up with at least one application of machine learning for image analysis that you think could have a net 
positive impact on the world.  For this application, please answer the following questions (we'll do a report out).
1. What is the application?  Who would derive value from it?
2. How would you validate that your system is working well enough to provide this value?
3. What would some potential (presumably unintended) negative consequences be of this application?
4. What might be some negative, off-label uses of this technology?  How could you mitigate the potential for these 
   negative consequences?
{% endcapture %} 
{% include problem.html problem=problem %}


# Preview of assignment and talking about convolution

We'll learn about convolutional neural networks to process images. First, we need to understand what a convolution means in this context.
[Assignment 17 - Images as Data and Convolutions](../assignments/assignment17/assignment17)