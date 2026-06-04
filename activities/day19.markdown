---
title: "Day 19: Convolutional Neural Networks and Privacy in ML" 
toc_sticky: true 
toc_h_max: 1
layout: problemset
published: true
---

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

# Data augmentation and image transformations

When training computer vision modules, data augmentation techniques are often uses to increase the diversity of 
training images.  I'll talk a little bit about the sorts of augmentations you might see, and then we can see [how you 
would do those in PyTorch](https://colab.research.google.com/drive/1lBUjxz5hJleKTt_zTrFwSyER0Er3wA8-?usp=sharing).

# Privacy in machine learning

There are a number of ways in which machine learning systems pose a risk to privacy.  Today, we're going to talk 
about the privacy of user data when interacting with machine learning systems.  Another class of privacy concerns, 
which we will not talk about today, is the role of machine learning systems in surveillance.

{% capture problem %}
Take a look at some of the materials in this section.  With respect to privacy risks associated with AI, were there 
ones that you found particularly surprising?  Which one feels like the most pressing to address as a society?  
With respect to the privacy frameworks (FIPPs and GDPR), are there particular considerations that you feel are too 
onerous?  Are there protections that feel particularly important? Start out by discussing at your table and we'll share out.
{% endcapture %}
{% include problem.html problem=problem %}

## Overview of Privacy Risks (and some regulatory frameworks)

IBM has a nice concise article [Exploring privacy issues in the age of AI](https://www.ibm.com/think/insights/ai-privacy)

## Fair Information Practice Principles (FIPPs)

One highly influential framework for data privacy is the Fair Information Practice Principles (FIPPs).  These 
principles were articulated at a 1980 at the [Convention for_the_Protection_of_Individuals_with_Regard_to_Automatic_Processing_of_Personal_Data](https://en.wikipedia.org/wiki/Convention_for_the_Protection_of_Individuals_with_Regard_to_Automatic_Processing_of_Personal_Data)

This convention eventually gave rise to [a treaty](https://rm.coe.int/1680078b37) that the member states of the 
Council of Europe ratified.  The [framework was updated in 2018](https://www.europarl.europa.eu/meetdocs/2014_2019/plmrep/COMMITTEES/LIBE/DV/2018/09-10/Convention_108_EN.pdf), largely in response to artificial intelligence.

The document sets out the following principles that govern privacy of data processing systems.

**Legitimacy of data processing and quality of data**
1. Data processing shall be proportionate in relation to the legitimate purpose pursued and reflect at all stages of 
   the processing a fair balance between all interests concerned, whether public or private, and the rights and 
   freedoms at stake.
2. Each Party shall provide that data processing can be carried out on the basis of the free, specific, informed and 
   unambiguous consent of the data subject or of some other legitimate basis laid down by law.
3. Personal data undergoing processing shall be processed lawfully.
4. Personal data undergoing processing shall be
   - processed fairly and in a transparent manner;
   - collected for explicit, specified and legitimate purposes and not processed in a way incompatible with those 
     purposes; further processing for archiving purposes in the public interest, scientific or historical research 
     purposes or statistical purposes is, subject to appropriate safeguards, compatible with thosepurposes;
   - adequate, relevant and not excessive in relation to the purposes for which they are processed;
   - accurate and, where necessary, kept up to date;
   - preserved in a form which permits identification of data subjects for no longer than is necessary for the 
     purposes for which those data are processed.

**Special Categories of Data**

1. The processing of: genetic data; personal data relating to offences, criminal proceedings and convictions, and 
related security measures;– biometric data uniquely identifying a person;– personal data for the information they 
reveal relating to racial or ethnic origin, political opinions, trade-union membership, religious or other beliefs, 
health or sexual life,shall only be allowed where appropriate safeguards are enshrined in law, complementing those 
of this Convention.
2. Such safeguards shall guard against the risks that the processing of sensitive data may present for the interests, 
   rights and fundamental freedoms of the data subject, notably a risk of discrimination.

**Transparency of processing**
1. Each Party shall provide that the controller informs the data subjects of: a. his or her identity and habitual 
residence or establishment; b. the legal basis and the purposes of the intended processing; c. the categories of 
personal data processed; d. the recipients or categories of recipients of the personal data, if any; and e. the means 
of exercising the rights set out in Article 9,
2. Paragraph 1 shall not apply where the data subject already has the relevant information.
3. Where the personal data are not collected from the data subjects, the controller shall not be required to provide 
   such information where the processing is expressly prescribed by law or this proves to be impossible or involves 
   disproportionate efforts.

**Article 9 – Rights of the data subject**

Every individual shall have a right:
1. not to be subject to a decision significantly affect-ing him or her based solely on an automated process-ing of 
data without having his or her views taken into consideration;
2. to obtain, on request, at reasonable intervals and without excessive delay or expense, confirmation of the 
processing of personal data relating to him or her, the communication in an intelligible form of the data processed, 
   all available information on their origin, on the preservation period as well as any other information that the 
   controller is required to provide in order to ensure the transparency of processing in accordance with Article 8, 
   paragraph 1;
3. to obtain, on request, knowledge of the reason-ing underlying data processing where the results of such processing 
   are applied to him or her;
4. to object at any time, on grounds relating to his or her situation, to the processing of personal data concerning 
   him or her unless the controller demonstrates legitimate grounds for the processing which override his or her 
   interests or rights and fundamental freedoms;
5. to obtain, on request, free of charge and with-out excessive delay, rectification or erasure, as the case may be, 
   of such data if these are being, or have been, processed contrary to the provisions of this Convention;
6. to have a remedy under Article 12 where his other rights under this Convention have been violated;
7. to benefit, whatever his or her nationality or residence, from the assistance of a supervisory authority within 
   the meaning of Article 15, in exercising his orher rights under this Convention.

## GDPR

The [GDPR](https://gdpr-info.eu/) (General Data Protection Regulation) is an EU regulatory framework to protect personal data and privacy.  
It is governed by largely the same principles as the FIPPs, but focuses more on regulation methods and frameworks 
rather than solely articulating broad principles (although you can take a look at the [specific principles outlined](https://gdpr-info.eu/chapter-2/))

As it relates to AI, one important part of GDPR is [article 22](https://gdpr-info.eu/art-22-gdpr/)

> Article 22 provides protections to individuals against
> decisions "based solely on automated processing" of
> personal data without human intervention, also called
> automated decision-making (ADM).24 It enshrines
> the right of individuals not to be subject to ADM
> where these decisions could produce an adverse
> legal or similarly significant effect on them. Given the
> widespread use of ADM as it relates to health, loan
> approvals, job applications, law enforcement, and
> other fields, the article plays a crucial role in enforcing
> a minimum degree of human involvement in such
> decision-making processes

## Allowances for AI training and development

There have been some (controversial) efforts to streamline GDPR compliance for AI companies.  Here is an [article 
critiquing the effort](https://www.amnesty.org/en/latest/news/2026/04/eu-simplification-laws/).

## Privacy Case Study

Technology for people with disabilities has been an area where there has often been a tension between the privacy 
concerns of individual users and the need for companies to either monetize or explicitly tune their systems to user 
data.  One relevant case study is [AIRA](https://aira.io/), which is an on-demand visual interpretation service.  
They articulate their [privacy policy](https://aira.io/aira-explorer-privacy-policy/) in this document.

Ethan Smith, who was one of the six students who worked on the EchoMinds project over the summer, recorded this 
video describing how he thought about the privacy / functionality tradeoffs of joining [AIRA's trusted tester program](https://aira.io/introducing-project-astra-join-the-trusted-testers-waitlist-today/) 
that came with significant privacy tradeoffs.

Let's [watch Ethan's video together](https://olincollege.sharepoint.com/:v:/s/EducateAI/IQDQFjbyXRu2Q7CYZkkWa9AuAdGooBxWyNRZnmU-F0HRlXM?e=ejRtgS&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D) (not publicly available currently)

Some discussion questions:
1. If you were in Ethan's shoes, would you trade your personal data for access to AIRA's service?
2. Do you think what AIRA is doing is ethical?
3. If you were developing your own technology for people who are blind, how would you handle use privacy?

# Next assignment preview

We'll discuss the next assignment and get started.
