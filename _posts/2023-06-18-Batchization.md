---
layout: post
published: True
title: Batches with texts of different lengths
---

When I was experimenting with [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy I saw that as in many other machine learning training, it's common to concatenate the sample of the dataset separating them with an <end-of-sequence> token. This makes sense to me at the moment when your model has to learn the human language structure but not during the fine-tuning process. This is because we could concatenate texts with completely different and uncorrelated contexts inducing the model to evaluate the generation of the next token based on, possibly, two different topics.

To solve this I tried to create a simple batch pipeline called batchization process which groups the texts by the number of tokens plus some tricks.
Let's check this out!

##### High-level guideline:
For clarity, we will divide the process into two steps
- Grouping the texts by length
- Creating a pseudo dataloader

<!--more-->

## Clusterize the samples by length

The initial idea is to group the samples by the number of tokens.
In particular, here, the __length_clusterization__ function takes in input a dataset which should be iterable of lists, where each list contains a tokenized text.
The function creates a dictionary where the key is the number of tokens in each group and the value is a list containing all the samples of that length.

```python
def length_clusterization(dataset):
    model_context = 1024
    cluster = {}
    
    for data in dataset:
        length = len(data)
        if length not in cluster:
            cluster[length] = []
        # Crop the length to model_context
        temp = data if length <= model_context else data[-model_context:]
        cluster[length].append(temp)

    # Sort by keys
    cluster = dict(sorted(cluster.items()))

    return cluster
```

## Create a pseudo dataloader

Here we create a dataset that is a dataloader because we check if the number of elements in each of the dictionary cluster keys is a multiple of the __batch_size__
we concatenate the values into the final list while if the reminder is not zero before concatenating we crop the surplus elements.

```python
def batchization(cluster, batch_size):
    final = []
    for key in list(cluster):
        if len(cluster[key]) % batch_size != 0 and len(cluster[key]) >= batch_size:
            final += cluster[key][:-(len(cluster[key]) % batch_size)]
        elif len(cluster[key]) % batch_size == 0 and len(cluster[key]) >= batch_size:
            final += cluster[key]

    return final
```

## Conclusions
The final list returned by the batchization function contains elements that are of different lengths but grouped by len in multiples of the batch_size.

As you can see the process is straightforward.
Here are some concerns:
- Is not clear if it will improve the final fine-tuning performance of your model
- It can become computationally heavy if the dataset is too large
- Improvements in computational cost can be applied
- Concatenation can be improved taking into account also the elements discarded in the case of non-zero reminder. For example by popping random tokens and clustering again.

I hope you have found this guide helpful. 

Please hit me up on <a href="https://twitter.com/Valeman100">Twitter</a> for any corrections or feedback.I hope you have found this guide helpful. 
Everything is ready to train the LLM beast model you desire.
