---
layout: post
published: False
title: How to connect Lambdalabs to Pycharm via SSH
---

In this article, we are going to see how to connect the famous editor by  [JetBrains Pycharm](https://www.jetbrains.com/pycharm/promo/?source=google&medium=cpc&campaign=14123077402&term=pycharm&content=536947779993&gad=1) with the cheapest cloud GPU provider nowadays [Lambda Cloud](https://lambdalabs.com/).

For the ones new to the topic, GPU cloud services permit you to run your scripts on powerful machines with a lot of RAM and the computational performances of GPUs. In particular, Lambdalabs provides a cheap, fast, and reliable service.

##### High-level guideline:
- Instantiate a machine on lambdalabs
- open the ssh via terminal and setup the environment
- link pycharm


<!--more-->

## Instantiate a GPU cloud machine
Let's begin by running a machine. This can be easily done after you created your account and linked the payment method (don't worry they will only charge for what you consume). Now that you are logged in you should see something similar to the picture below.

<div class="img-div-any-width" markdown="0">
  <img src="/images/lambda/lambda_init.png" />
</div>
