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

## Preliminary step

First of all, we need a GPU machine because the quantization process requires CUDA drivers. 
Clone llama.cpp using:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
```

If we trained the model utilizing the huggingface packages or pytorch it's time to convert the model weights in a format called ggml and then from that checkpoint will be possible to generate the quantized transformer.

## Compile, Convert, Quantize

We will use a Python script inspired by this [issue](https://huggingface.co/junelee/wizard-vicuna-13b/discussions/2) to achieve our goal in the simplest way possible. Let's analyze the code:

```python
#!/usr/bin/env python3
import argparse
import os
import subprocess

def main(model, outbase, outdir):
    #path to llama.cpp local repo
    llamabase = "/your/path/llama.cpp"
    ggml_version = "v3"
    if not os.path.isdir(model):
        raise Exception(f"Could not find model dir at {model}")
    if not os.path.isfile(f"{model}/config.json"):
        raise Exception(f"Could not find config.json in {model}")
    
    os.makedirs(outdir, exist_ok=True)

    print("Building llama.cpp")
    subprocess.run(f"cd {llamabase} && git pull && make clean && LLAMA_CUBLAS=1 make", shell=True, check=True)
    subprocess.run(f"pip install sentencepiece", shell=True, check=True)

    fp16 = f"{outdir}/{outbase}.ggml{ggml_version}.fp16.bin"
    print(f"Making unquantised GGML at {fp16}")
    if not os.path.isfile(fp16):
        subprocess.run(f"python3 {llamabase}/convert.py {model} --outtype f16 --outfile {fp16}", shell=True, check=True)
    else:
        print(f"Unquantised GGML already exists at: {fp16}")

    print("Making quants")
    for type in ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]:
        outfile = f"{outdir}/{outbase}.ggml{ggml_version}.{type}.bin"
        print(f"Making {type} : {outfile}")
        subprocess.run(f"{llamabase}/quantize {fp16} {outfile} {type}", shell=True, check=True)

    # Delete FP16 GGML when done making quantizations
    os.remove(fp16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Bash to Python.')
    parser.add_argument('model', help='Your finetuned Model weight directory', default='light-study-tags-requirements', nargs='?')
    parser.add_argument('outbase', help='Output base name', default='llama.cpp',nargs='?')
    parser.add_argument('outdir', help='Output directory', default='quantized', nargs='?')

    args = parser.parse_args()

    main(args.model, args.outbase, args.outdir)
```

Let's go through the script:
- initially, we set the path to the llama.cpp repo and check that the model location we will pass later is a directory containing all the required staff. In particular, remember to put inside the model's folder the weights the config.json, and the tokenizer checkpoint.
- creates the output dir where to put the final result.
- a bash script will build the llama.cpp repo and run the convert.py script which transforms our pytorch checkpoint into a ggml_v3 model.
- finally, the quantize script creates four quantized models: "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"

For those curious about the meaning of the 4 quantized versions, [here](https://www.reddit.com/r/LocalLLaMA/comments/139yt87/notable_differences_between_q4_2_and_q5_1/) you can find an insight.

## Inference time
Let's wrap up the ideas. We executed convert.py to transform our fine-tuned llama model into another format compatible with the C++ implementation provided by llama.cpp.
Now we are ready to transfer the generated checkpoint in whichever platform we prefer and enjoy the generation on CPU, GPU, or MPS.

There are a couple of ways to do so:

1) first one is to run directly from the bash:
   
  _./main -m ./your-quantized-model-path/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -n 512_

    this script generates a completion for the prompt (-p inline command) that asks to generate a 10-step list about how to build a website, furthermore, we set the max length for the completion at 512 (-n).
    There are several other options we can set which you can view from your bash terminal directly.

3) The second option, which is more useful in my opinion, is to adopt bindings.
In llama.cpp repo there are already binders for Python, Go, Node.js, Ruby, C#/.NET. with them, you can instantiate a server or use a prebuilt docker image, build your own, or even use the quantized model in a Python script directly.

I hope you have found this guide helpful.

Please hit me up on <a href="https://twitter.com/Valeman100">Twitter</a> for any corrections or feedback.

## More Resources

* To understand quantization [high level](https://huggingface.co/docs/optimum/concept_guides/quantization).
