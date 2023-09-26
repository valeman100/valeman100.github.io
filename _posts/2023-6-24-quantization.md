---
layout: post
published: True
title: How to quantize your finetuned llama model
---

<div class="img-div-any-width" markdown="0">
  <img src="/images/quantized_llama.png" />
</div>

Imagine you have just trained your brand new large language model using a supercluster with 8xA100 80GB on multiple nodes but now find butterflies flying away from your pocket and you can infer your creation only on a low-budget CPU machine or simply you are looking for a cheap way to put in production your buddy.
In this guide, we will see how to shrink as much as we can the memory usage of our model and be able to run it with as small resources as 8GB of RAM. 

To reach the top we will exploit two tricks:
- int precision quantization
- C++ code conversion

All of this will be possible thanks to the amazing work of [__llama.cpp__](https://github.com/ggerganov/llama.cpp) !!!

###### Disclaimer
This guide has been tested with a finetuned version of llama 7B from the [huggingface hub](https://huggingface.co/huggyllama/llama-7b) which uses the Vicuna training pipeline but in general, should work with any llama model that is saved in a pytorch fashion.

##### High-level summary:

- Clone lama.cpp repo on a machine equipped with GPU.
- Compile the repo and Quantize your model.
- Enjoy inference from a terminal, web server, python, or docker on almost any device.

<!--more-->

Read the full article on [Medium[(https://medium.com/@val.mannucci/how-to-quantize-your-finetuned-llama-model-24e7b42c1ad6)

Please hit me up on <a href="https://twitter.com/Valeman100">Twitter</a> for any corrections or feedback.

## More Resources

* To understand quantization [high level](https://huggingface.co/docs/optimum/concept_guides/quantization).
