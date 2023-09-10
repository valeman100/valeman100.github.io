---
layout: post
published: False
title: Finetune and validate your version of ChatGPT
---

<div class="img-div-any-width" markdown="0">
  <img src="/images/reading_robot.png" />
</div>

Finally last week OpenAI released the APIs to finetune the popular model on which ChatGPT is based!

Following [their suggestions](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_finetune_chat_models.ipynb) we will go through step by step from the dataset preparation to the tuned model usage.

Until a few months ago finetuning a general model was not a straightforward task, it involved knowledge in training, ML frameworks, and the availability of computational units like GPUs.
Lately, things are becoming easier thanks to people's open source projects which already implement the heavy parts for you.

OpenAI gives you the chance to train your LLM even if you don't know anything about the subject, you'll just need a bit of Python and an OpenAI account.

##### What we will cover:
- When and why adopt finetune
- Dataset preparation
- Finetune
- Validation
- Final thoughts

<!--more-->

## Do you need to finetune?

A times this question is underrated but in reality, it's a proper concern to open before embarking on the finetuning way.

Go for finetune only if:

1. You drained all your prompt engineering skills. First, you have to try different prompts and refactorizations to see if the answers improve.
2. Try few-shot prompting. Add some examples to your prompt to provide the model with some more context on which to base its answer.

If you are interested I suggest reading [this article](https://www.tidepool.so/2023/08/17/why-you-probably-dont-need-to-fine-tune-an-llm/?utm_source=tldrai) which goes in-depth with the concepts.

## Prepare your Data

The main course of the supper is the goodness of the dataset. It must be:
- informative, i.e. the examples need to have the prompts that give you the best answer without finetune,
- the ground truth completion we will use to tune the model must be unique for each example,
- correctly formatted.

While for the first points I can't help you, with the last one I show you the correct packing for your data.
For your convenience is useful to run the code in a jupyter notebook also because a times the APIs take minutes to complete the processes.

```python
import json
import openai
import os
from pprint import pprint
from dotenv import load_dotenv

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY", "your key")

def get_dataset():
    my_data = pd.read_csv('data.csv')  

    data = []

    for idx, conversation in my_data.iterrows():
        data.append({"messages": [
            {"role": "system", "content": conversation['prompt']['system']},
            {"role": "user", "content": conversation['prompt']['user']},
            {"role": "assistant", "content": conversation['completion']}]})

    return data

data = get_dataset()
print(len(data))
th = len(data)*0.9

training_data = data[:th]
validation_data = data[th:]
```

OpenAI APIs require your data to follow the format above. Here I suppose your dataset is in a pandas dataframe format. We refactor it into a list containing the examples of your training and then we divide it into training (90%) and validation (10%)

The official guide suggests as few as 50-100 examples to already reach satisfactory performances but don't be scared to experiment the finetuning is kind of cheap (we will calculate the price further on).

Remember to set your OpenAI key in your env or pass it at the beginning of the code.

We are ready to send the data to OpenAI servers.

```python
def write_jsonl(data_list: list, filename: str) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)

training_file_name = "finetune_training.jsonl"
write_jsonl(training_data, training_file_name)

validation_file_name = "finetune_validation.jsonl"
write_jsonl(validation_data, validation_file_name)

training_response = openai.File.create(
    file=open(training_file_name, "rb"), purpose="fine-tune"
)
training_file_id = training_response["id"]

validation_response = openai.File.create(
    file=open(validation_file_name, "rb"), purpose="fine-tune"
)
validation_file_id = validation_response["id"]

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)
```

The function 'write_jsonl' writes the list in a jsonl file, which is sent through the create API to your OpenAI cloud.

We can monitor the loading process using the ID and the following call.

```python
print (openai.File.retrieve(training_file_id), openai.File.retrieve(validation_file_id))
```
It may take some minutes but when the status goes under "processed" means we are ready for the next step which involves true real finetune.

## Let's Finetune

Create a finetuning job passing the dataset IDs, one of the available tunable models (I advise you to choose 'gpt-3.5-turbo' i.e. the one used for ChatGPT, it's also the latest tunable released), a suffix for your job. It's also possible to set the number of epochs which by default are set to 3.

```python
response = openai.FineTuningJob.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-3.5-turbo",
    suffix="my-tune",
)

job_id = response["id"]

print("Job ID:", response["id"])
print("Status:", response["status"])
print("Trained Tokens:", response["trained_tokens"])
print("Price:", round(response["trained_tokens"]*0.008*0.0001*response["hyperparameters"]['n_epochs'], 2), '$')
```

Finally, we printed some useful info which include also the prices (referred to the specific model we selected).
In this case, if we want to calculate the cost in advance the formula is:
$$finetuning cost = \frac{training tokens x n_epochs}{1000 x 0.008\$}$$

which for a training file with 100,000 tokens trained over 3 epochs, the expected cost would be ~ $2.40.

To check the status and the loss of our job we can run the following cell:

```python
response = openai.FineTuningJob.list_events(id=job_id, limit=50)

events = response["data"]
events.reverse()

for event in events:
    print(event["message"])
```

At the end of the training, we are ready to store the model ID that we'll use to call the classic OpenAI completion API.

```python
response = openai.FineTuningJob.retrieve(job_id)
fine_tuned_model_id = response["fine_tuned_model"]
print("Fine-tuned model ID:", fine_tuned_model_id)
```

## Flex your new model
Calling the new model is as simple as this.

```python
response = openai.ChatCompletion.create(
    model=fine_tuned_model_id, messages=your_message, temperature=0, max_tokens=1000
)
print(response["choices"][0]["message"]["content"])
```

Remember to format your prompt as we did when we prepared the dataset with system (optional) and user in separate dictionaries.

## Validation

At this point, we are curious to see if the training has improved the output performances.

Here I provide a simple function that permits you to print the outputs from different models using the same prompt.
You have just to define a list of model names you are interested in (made available by OpenAI) and set a ground truth answer, if you have one. 

For sure our first option is to validate against the non-finetuned model.

```python
import json
import os
from typing import List
import openai
from dotenv import load_dotenv
from pprint import pprint

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY", "your key")


def evaluate_against(models: List, prompt: str, ground_truth=None):
    mod = openai.Model.list()
    model_list = [m['id'] for m in mod['data']]

    for m in models:
        if m not in model_list:
            print(f"{m} not in models' list")
            print(f'Available models: {model_list}')
            return

    if ground_truth:
        print(f'\n--- Ground truth ---')
        pprint(ground_truth)

    for m in models:
        response = openai.ChatCompletion.create(
            model=m, messages=prompt, temperature=0, max_tokens=1000
        )
        print(f'--- Model: {m} ---')
        pprint(response["choices"][0]["message"]["content"])


if __name__ == '__main__':
    with open('data.json', 'r') as f:
        data = json.load(f)

    models = ['your_model_ID', 'gpt-3.5-turbo-0613']
    for ex in data:
        evaluate_against(models, ex['messages'][:2], ex['messages'][2:])
```

## Final Thoughts

In this guide, we saw how to finetune the most powerful model currently available by OpenAI using as little as a few lines of code.
Remember that this process is not always the holy grail for your problems, in particular, is extremely effective in domain adaptation problems (to teach the style of your outputs).
For tasks that require increasing the knowledge of the model, I would suggest using a RAG.

I hope you have found this guide helpful. 

Please hit me up on <a href="https://twitter.com/Valeman100">Twitter</a> for any correction or feedback.
