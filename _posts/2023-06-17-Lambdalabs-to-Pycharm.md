---
layout: prediction_post
published: True
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

The next step is to add your public ssh key. It is as simple as clicking on the __ssh keys__ on the left and adding it. If you don't know where to find it on your local machine it is usually under the path you see below. Try this script in your terminal:

```bash
cd /Users/your-machine-name/.ssh
cat id_rsa.pub 
```

Now you are ready to instantiate your machine. Click on __launch instance__ and select the one you prefer. Wait until the boot is finished (should take a couple of minutes) then copy the __ssh login__ field.

<div class="img-div-any-width" markdown="0">
  <img src="/images/lambda/lambda_instantiated.png" />
</div>

## Connect via terminal

This step is required because the virtual environment of pycharm goes in conflict with the one you find installed inside the Lambdalab machines.
All you have to do is to launch a new terminal window and run:

```bash
ssh ubuntu@192.9.128.220
sudo apt remove -y python3-virtualenv
pip install --user virtualenv --force-reinstall
```

ssh will ask you to add the ip inside the local list of your IP addresses, say yes and proceed. Now you are inside the machine terminal. The second and third commands go there.

Note: if the second command (sudo ...) gets stuck don't worry kill the process and proceed with the last one.

## Connect Pycharm

Open your pycharm project and on the bottom right you can find the remote interpreter. Otherwise go in __settings__ > __python interpreter__ and select __add new interpreter__ > __on ssh__

Compile the panel as shown in the pic below. Remember to put the __IP__ of the machine you started on lambda.

<div class="img-div-any-width" markdown="0">
  <img src="/images/lambda/pycharm_init.png" />
</div>

then just click next till the virtual env setup, which must be compiled as follow:

<div class="img-div-any-width" markdown="0">
  <img src="/images/lambda/pychar_env.png" />
</div>

This will automatically upload your whole project on the ssh machine. If you have heavy files like model checkpoints I recommend stopping it and excluding them from the automatic upload. It is faster to upload them via sftp or from huggingface_hub or directly from aws if you have saved them there.

I hope you have found this guide helpful. 
Everything is ready to train the LLM beast model you desire.

Please hit me up on <a href="https://twitter.com/Valeman100">Twitter</a> for any corrections or feedback.
