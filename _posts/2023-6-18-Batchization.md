---
layout: post
published: True
title: Batches with texts of different lengths
---

<div class="img-div-any-width" markdown="0">
  <img src="/images/long_short.png" />
</div>

When I was experimenting with [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy I saw that as in many other machine learning training, it's common to concatenate the sample of the dataset separating them with an <end-of-sequence> token. This makes sense to me at the moment when your model has to learn the human language structure but not during the fine-tuning process. This is because we could concatenate texts with completely different and uncorrelated contexts inducing the model to evaluate the generation of the next token based on, possibly, two different topics.

To solve this I tried to create a simple batch pipeline called batchization process which groups the texts by the number of tokens plus some tricks.
Let's check this out!

##### High-level guideline:
For clarity, we will divide the process into two steps
- Grouping the texts by length
- Creating a pseudo dataloader

<!--more-->

Read the full article on [Medium](https://medium.com/@val.mannucci/batches-with-texts-of-different-lengths-343e4a506fcd)

Please hit me up on <a href="https://twitter.com/Valeman100">Twitter</a> for any corrections or feedback.I hope you have found this guide helpful. 
Everything is ready to train the LLM beast model you desire.
