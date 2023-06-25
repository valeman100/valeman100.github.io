---
layout: prediction_post
published: True
title: Il transformer illustrato - IT
---

##### Disclaimer
Traduzione italiana di [The illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
Non sono un traduttore professionista. 
La proprietà intellettuale dell'articolo è di [Jay Alammar](http://jalammar.github.io/illustrated-transformer/)

Italian translation of [The illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
I'm not a professional translator. 
The intellectual property of the article is owned by [Jay Alammar](http://jalammar.github.io/illustrated-transformer/)

Nel [post precedente, abbiamo esaminato l'Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) -- un metodo onnipresente nei moderni modelli di deep learning. L'attention è un concetto che ha contribuito a migliorare le prestazioni delle applicazioni di traduzione automatica con modelli neurali. In questo post, esamineremo **The Transformer**, un modello che utilizza l'attenzione per aumentare la velocità con cui questi modelli possono essere addestrati. Il Trasformer supera il modello di traduzione automatica neurale di Google in attività specifiche. Il più grande vantaggio, tuttavia, deriva dal modo in cui il Transformer si presta alla parallelizzazione. È infatti raccomandazione di Google Cloud utilizzare il Transformer come modello di riferimento per utilizzare la loro proposte di [Cloud TPU](https://cloud.google.com/tpu/). Quindi proviamo a scomporre il modello e vediamo come funziona.

Il Transformer è stato proposto nell'articolo [Attention is All You Need](https://arxiv.org/abs/1706.03762). Una sua implementazione TensorFlow è disponibile come parte del pacchetto [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor). Il gruppo NLP di Harvard ha creato una [guida che spiega l'articolo con l'implementazione di PyTorch](http://nlp.seas.harvard.edu/2018/04/03/attention.html). In questo post, cercheremo di semplificare un po' le cose e di introdurre i concetti uno per uno, sperando che sia più facile da capire per le persone senza una conoscenza approfondita dell'argomento.

**Aggiornamento 2020**: Ho creato il video "Transformer narrati" che è un approccio meno impegnativo all'argomento:

 <div style="text-align:center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/-QH8fRhqFHM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"  style="
 width: 100%;
 max-width: 560px;"
allowfullscreen></iframe>
</div>
## Uno sguardo generale
Iniziamo osservando il modello come una singola scatola nera. In un'applicazione di traduzione automatica, prenderebbe una frase in una lingua e restituirebbe la sua traduzione in un'altra.


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/the_transformer_3.png" />
</div>


<!--more-->


Aprendo quella meraviglia di Optimus Prime, vediamo un componente di codifica, un componente di decodifica e le connessioni tra di essi.

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/The_transformer_encoders_decoders.png" />
</div>

Il componente di codifica è uno stack di encoders (il paper ne concatena sei uno dietro l'altro - non c'è nulla di magico nel numero sei, si possono sicuramente fare esperimenti con altre configurazioni). Il componente di decodifica è una concatenazione di decoders dello stesso numero.

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/The_transformer_encoder_decoder_stack.png" />
</div>

Gli encoders sono tutti identici nella struttura (ma non condividono i pesi). Ognuno è suddiviso in due sottolivelli:


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/Transformer_encoder.png" />
</div>

Gli input dell'encoder passano prima attraverso uno strato di self-attention, uno strato che aiuta il codificatore a guardare altre parole nella frase di input mentre codifica una parola specifica. Analizzeremo più da vicino la self-attention in seguito nel post.

Le uscite dello strato di self-attention vengono inviate a una rete neurale feed-forward. La stessa rete feed-forward viene applicata indipendentemente a ogni posizione.

Il decoder ha entrambi quei livelli, ma tra di essi c'è uno strato di attention che aiuta il decoder a focalizzarsi sulle parti rilevanti della frase di input (simile a ciò che fa l'attention nei [modelli seq2seq](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)).

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/Transformer_decoder.png" />
</div>


## Introduzione dei tensori al meccanismo

Ora che abbiamo visto i principali componenti del modello, iniziamo a esaminare i vari vettori/tensori e come fluiscono tra questi componenti per trasformare l'input di un modello addestrato in un output.

Come avviene nelle applicazioni di NLP in generale, iniziamo trasformando ogni parola di input in un vettore utilizzando un [algoritmo di embedding](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca).

<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/embeddings.png" />
  <br />
  Ogni parola viene convertita in un vettore di dimensione 512. Rappresenteremo questi vettori con questi semplici riquadri.
</div>

L'embedding avviene solo nell'encoder più in basso. L'astrazione comune a tutti gli endoders è che ricevono una lista di vettori, ognuno di dimensione 512-- Nell'encoder più in basso l'input sarebbe direttamente l'output dell'algoritmo di embedding, mentre negli altri encoders sarebbe l'output dell'encoder sottostante. La dimensione di questa lista è un iperparametro che possiamo impostare - fondamentalmente corrisponderà alla lunghezza della frase più lunga nel nostro set di dati di addestramento.

Dopo l'embedding delle parole della nostra sequenza di iniziale, ognuna di esse viene processata da ciascuno dei due strati dell'encoder.



<div class="img-div-any-width" markdown="0">
  <img src="/images/t/encoder_with_tensors.png" />
  <br />

</div>

Qui iniziamo a vedere una caratteristica chiave del Transformer, ovvero che ogni parola fluisce lungo il proprio percorso nell'encoder. Ci sono dipendenze tra questi percorsi nello strato di self-attention. Lo strato feed-forward, tuttavia, non ha tali dipendenze, e quindi i vari percorsi possono essere eseguiti in parallelo mentre passano attraverso lo strato feed-forward.

Successivamente, cambieremo l'esempio con una frase più breve e osserveremo cosa accade in ciascun sottolivello del codificatore.

## Now We're Encoding!

As we've mentioned already, an encoder receives a list of vectors as input. It processes this list by passing these vectors into a 'self-attention' layer, then into a feed-forward neural network, then sends out the output upwards to the next encoder.


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/encoder_with_tensors_2.png" />
  <br />
  The word at each position passes through a self-attention process. Then, they each pass through a feed-forward neural network -- the exact same network with each vector flowing through it separately.
</div>

## Self-Attention at a High Level
Don't be fooled by me throwing around the word "self-attention" like it's a concept everyone should be familiar with. I had personally never came across the concept until reading the Attention is All You Need paper. Let us distill how it works.

Say the following sentence is an input sentence we want to translate:

"```The animal didn't cross the street because it was too tired```"

What does "it" in this sentence refer to? Is it referring to the street or to the animal? It's a simple question to a human, but not as simple to an algorithm.

When the model is processing the word "it", self-attention allows it to associate "it" with "animal".

As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.

If you're familiar with RNNs, think of how maintaining a hidden state allows an RNN to incorporate its representation of previous words/vectors it has processed with the current one it's processing. Self-attention is the method the Transformer uses to bake the "understanding" of other relevant words into the one we're currently processing.


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self-attention_visualization.png" />
  <br />
  As we are encoding the word "it" in encoder #5 (the top encoder in the stack), part of the attention mechanism was focusing on "The Animal", and baked a part of its representation into the encoding of "it".
</div>


Be sure to check out the [Tensor2Tensor notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) where you can load a Transformer model, and examine it using this interactive visualization.



## Self-Attention in Detail
Let's first look at how to calculate self-attention using vectors, then proceed to look at how it's actually implemented -- using matrices.

The **first step** in calculating self-attention is to create three vectors from each of the encoder's input vectors (in this case, the embedding of each word). So for each word, we create a Query vector, a Key vector, and a Value vector. These vectors are created by multiplying the embedding by three matrices that we trained during the training process.

Notice that these new vectors are smaller in dimension than the embedding vector. Their dimensionality is 64, while the embedding and encoder input/output vectors have dimensionality of 512. They don't HAVE to be smaller, this is an architecture choice to make the computation of multiheaded attention (mostly) constant.


<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self_attention_vectors.png" />
  <br />
  Multiplying <span class="encoder">x1</span> by the <span class="decoder">WQ</span> weight matrix produces <span class="decoder">q1</span>, the "query" vector associated with that word. We end up creating a "query", a "key", and a "value" projection of each word in the input sentence.
</div>

<br />
<br />

What are the "query", "key", and "value" vectors?
<br />
<br />
They're abstractions that are useful for calculating and thinking about attention. Once you proceed with reading how attention is calculated below, you'll know pretty much all you need to know about the role each of these vectors plays.

The **second step** in calculating self-attention is to calculate a score. Say we're calculating the self-attention for the first word in this example, "Thinking". We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.

The score is calculated by taking the dot product of the <span class="decoder">query vector</span> with the <span class="context">key vector</span> of the respective word we're scoring. So if we're processing the self-attention for the word in position <span class="encoder">#1</span>, the first score would be the dot product of <span class="decoder">q1</span> and <span class="context">k1</span>. The second score would be the dot product of <span class="decoder">q1</span> and <span class="context">k2</span>.


<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self_attention_score.png" />
  <br />

</div>

<br />


The **third and fourth steps** are to divide the scores by 8 (the square root of the dimension of the key vectors used in the paper -- 64. This leads to having more stable gradients. There could be other possible values here, but this is the default), then pass the result through a softmax operation. Softmax normalizes the scores so they're all positive and add up to 1.


<br />


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/self-attention_softmax.png" />
  <br />

</div>

This softmax score determines how much each word will be expressed at this position. Clearly the word at this position will have the highest softmax score, but sometimes it's useful to attend to another word that is relevant to the current word.

<br />


The **fifth step** is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example).

The **sixth step** is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).

<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/self-attention-output.png" />
  <br />
</div>

That concludes the self-attention calculation. The resulting vector is one we can send along to the feed-forward neural network. In the actual implementation, however, this calculation is done in matrix form for faster processing. So let's look at that now that we've seen the intuition of the calculation on the word level.


## Matrix Calculation of Self-Attention
**The first step** is to calculate the Query, Key, and Value matrices. We do that by packing our embeddings into a matrix <span class="encoder">X</span>, and multiplying it by the weight matrices we've trained (<span class="decoder">WQ</span>, <span class="context">WK</span>, <span class="step_no">WV</span>).

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/self-attention-matrix-calculation.png" />
  <br />
  Every row in the <span class="encoder">X</span> matrix corresponds to a word in the input sentence. We again see the difference in size of the embedding vector (512, or 4 boxes in the figure), and the q/k/v vectors (64, or 3 boxes in the figure)
</div>

<br />

**Finally**, since we're dealing with matrices, we can condense steps two through six in one formula to calculate the outputs of the self-attention layer.

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/self-attention-matrix-calculation-2.png" />
  <br />
  The self-attention calculation in matrix form
</div>

<br />


<br />

## The Beast With Many Heads

The paper further refined the self-attention layer by adding a mechanism called "multi-headed" attention. This improves the performance of the attention layer in two ways:

 1. It expands the model's ability to focus on different positions. Yes, in the example above, z1 contains a little bit of every other encoding, but it could be dominated by the actual word itself. If we're translating a sentence like "The animal didn't cross the street because it was too tired", it would be useful to know which word "it" refers to.

 1. It gives the attention layer multiple "representation subspaces". As we'll see next, with multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.


 <div class="img-div-any-width" markdown="0">
   <img src="/images/t/transformer_attention_heads_qkv.png" />
   <br />
   With multi-headed attention, we maintain separate Q/K/V weight matrices for each head resulting in different Q/K/V matrices. As we did before, we multiply X by the WQ/WK/WV matrices to produce Q/K/V matrices.
 </div>

 <br />
If we do the same self-attention calculation we outlined above, just eight different times with different weight matrices, we end up with eight different Z matrices

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_attention_heads_z.png" />
  <br />

</div>

 <br />

This leaves us with a bit of a challenge. The feed-forward layer is not expecting eight matrices -- it's expecting a single matrix (a vector for each word). So we need a way to condense these eight down into a single matrix.

How do we do that? We concat the matrices then multiply them by an additional weights matrix WO.


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_attention_heads_weight_matrix_o.png" />
  <br />

</div>

That's pretty much all there is to multi-headed self-attention. It's quite a handful of matrices, I realize. Let me try to put them all in one visual so we can look at them in one place

<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_multi-headed_self-attention-recap.png" />
  <br />

</div>

<br />

Now that we have touched upon attention heads, let's revisit our example from before to see where the different attention heads are focusing as we encode the word "it" in our example sentence:

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self-attention_visualization_2.png" />
  <br />
  As we encode the word "it", one attention head is focusing most on "the animal", while another is focusing on "tired" -- in a sense, the model's representation of the word "it" bakes in some of the representation of both "animal" and "tired".
</div>

<br />

If we add all the attention heads to the picture, however, things can be harder to interpret:

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self-attention_visualization_3.png" />
  <br />
</div>



## Representing The Order of The Sequence Using Positional Encoding
One thing that's missing from the model as we have described it so far is a way to account for the order of the words in the input sequence.

To address this, the transformer adds a vector to each input embedding. These vectors follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence. The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they're projected into Q/K/V vectors and during dot-product attention.

<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_positional_encoding_vectors.png" />
  <br />
  To give the model a sense of the order of the words, we add positional encoding vectors -- the values of which follow a specific pattern.
</div>
  <br />


If we assumed the embedding has a dimensionality of 4, the actual positional encodings would look like this:

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_positional_encoding_example.png" />
  <br />
  A real example of positional encoding with a toy embedding size of 4
</div>

  <br />

What might this pattern look like?

In the following figure, each row corresponds to a positional encoding of a vector. So the first row would be the vector we'd add to the embedding of the first word in an input sequence. Each row contains 512 values -- each with a value between 1 and -1. We've color-coded them so the pattern is visible.



<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_positional_encoding_large_example.png" />
  <br />
  A real example of positional encoding for 20 words (rows) with an embedding size of 512 (columns). You can see that it appears split in half down the center. That's because the values of the left half are generated by one function (which uses sine), and the right half is generated by another function (which uses cosine). They're then concatenated to form each of the positional encoding vectors.
</div>


The formula for positional encoding is described in the paper (section 3.5). You can see the code for generating positional encodings in [```get_timing_signal_1d()```](https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py). This is not the only possible method for positional encoding. It, however, gives the advantage of being able to scale to unseen lengths of sequences (e.g. if our trained model is asked to translate a sentence longer than any of those in our training set).

**July 2020 Update:** 
The positional encoding shown above is from the Tranformer2Transformer implementation of the Transformer. The method shown in the paper is slightly different in that it doesn't directly concatenate, but interweaves the two signals. The following figure shows what that looks like. [Here's the code to generate it](https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb):

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/attention-is-all-you-need-positional-encoding.png" />
  <br />
</div>

## The Residuals
One detail in the architecture of the encoder that we need to mention before moving on, is that each sub-layer (self-attention, ffnn) in each encoder has a residual connection around it, and is followed by a [layer-normalization](https://arxiv.org/abs/1607.06450) step.

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_resideual_layer_norm.png" />
  <br />
</div>

If we're to visualize the vectors and the layer-norm operation associated with self attention, it would look like this:

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_resideual_layer_norm_2.png" />
  <br />
</div>

This goes for the sub-layers of the decoder as well. If we're to think of a Transformer of 2 stacked encoders and decoders, it would look something like this:

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_resideual_layer_norm_3.png" />
  <br />
</div>


## The Decoder Side
Now that we've covered most of the concepts on the encoder side, we basically know how the components of decoders work as well. But let's take a look at how they work together.

The encoder start by processing the input sequence. The output of the top encoder is then transformed into a set of attention vectors K and V. These are to be used by each decoder in its "encoder-decoder attention" layer which helps the decoder focus on appropriate places in the input sequence:


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_decoding_1.gif" />
  <br />
  After finishing the encoding phase, we begin the decoding phase. Each step in the decoding phase outputs an element from the output sequence (the English translation sentence in this case).
</div>

The following steps repeat the process until a special <end of sentence> symbol is reached indicating the transformer decoder has completed its output. The output of each step is fed to the bottom decoder in the next time step, and the decoders bubble up their decoding results just like the encoders did. And just like we did with the encoder inputs, we embed and add positional encoding to those decoder inputs to indicate the position of each word.

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_decoding_2.gif" />
  <br />

</div>

The self attention layers in the decoder operate in a slightly different way than the one in the encoder:

In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking future positions (setting them to ```-inf```) before the softmax step in the self-attention calculation.

The "Encoder-Decoder Attention" layer works just like multiheaded self-attention, except it creates its Queries matrix from the layer below it, and takes the Keys and Values matrix from the output of the encoder stack.

## The Final Linear and Softmax Layer

The decoder stack outputs a vector of floats. How do we turn that into a word? That's the job of the final Linear layer which is followed by a Softmax Layer.

The Linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders, into a much, much larger vector called a logits vector.

Let's assume that our model knows 10,000 unique English words (our model's "output vocabulary") that it's learned from its training dataset. This would make the logits vector 10,000 cells wide -- each cell corresponding to the score of a unique word. That is how we interpret the output of the model followed by the Linear layer.

The softmax layer then turns those scores into probabilities (all positive, all add up to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.

  <br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_decoder_output_softmax.png" />
  <br />
  This figure starts from the bottom with the vector produced as the output of the decoder stack. It is then turned into an output word.
</div>

  <br />

## Recap Of Training
Now that we've covered the entire forward-pass process through a trained Transformer, it would be useful to glance at the intuition of training the model.

During training, an untrained model would go through the exact same forward pass. But since we are training it on a labeled training dataset, we can compare its output with the actual correct output.

 To visualize this, let's assume our output vocabulary only contains six words("a", "am", "i", "thanks", "student", and "\<eos\>" (short for 'end of sentence')).

 <div class="img-div" markdown="0">
   <img src="/images/t/vocabulary.png" />
   <br />
   The output vocabulary of our model is created in the preprocessing phase before we even begin training.
 </div>

Once we define our output vocabulary, we can use a vector of the same width to indicate each word in our vocabulary. This also known as one-hot encoding. So for example, we can indicate the word "am" using the following vector:

<div class="img-div" markdown="0">
  <img src="/images/t/one-hot-vocabulary-example.png" />
  <br />
  Example: one-hot encoding of our output vocabulary
</div>

Following this recap, let's discuss the model's loss function -- the metric we are optimizing during the training phase to lead up to a trained and hopefully amazingly accurate model.

## The Loss Function
Say we are training our model. Say it's our first step in the training phase, and we're training it on a simple example -- translating "merci" into "thanks".

What this means, is that we want the output to be a probability distribution indicating the word "thanks". But since this model is not yet trained, that's unlikely to happen just yet.

<div class="img-div" markdown="0">
  <img src="/images/t/transformer_logits_output_and_label.png" />
  <br />
  Since the model's parameters (weights) are all initialized randomly, the (untrained) model produces a probability distribution with arbitrary values for each cell/word. We can compare it with the actual output, then tweak all the model's weights using backpropagation to make the output closer to the desired output.
</div>

<br />

How do you compare two probability distributions? We simply subtract one from the other. For more details, look at  [cross-entropy](https://colah.github.io/posts/2015-09-Visual-Information/) and [Kullback–Leibler divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained).

But note that this is an oversimplified example. More realistically, we'll use a sentence longer than one word. For example -- input: "je suis étudiant" and expected output: "i am a student". What this really means, is that we want our model to successively output probability distributions where:

 * Each probability distribution is represented by a vector of width vocab_size (6 in our toy example, but more realistically a number like 30,000 or 50,000)
 * The first probability distribution has the highest probability at the cell associated with the word "i"
 * The second probability distribution has the highest probability at the cell associated with the word "am"
 * And so on, until the fifth output distribution indicates '```<end of sentence>```' symbol, which also has a cell associated with it from the 10,000 element vocabulary.


 <div class="img-div" markdown="0">
   <img src="/images/t/output_target_probability_distributions.png" />
   <br />
   The targeted probability distributions we'll train our model against in the training example for one sample sentence.
 </div>

<br />

After training the model for enough time on a large enough dataset, we would hope the produced probability distributions would look like this:

  <div class="img-div" markdown="0">
    <img src="/images/t/output_trained_model_probability_distributions.png" />
    <br />
    Hopefully upon training, the model would output the right translation we expect. Of course it's no real indication if this phrase was part of the training dataset (see: <a href="https://www.youtube.com/watch?v=TIgfjmp-4BA">cross validation</a>). Notice that every position gets a little bit of probability even if it's unlikely to be the output of that time step -- that's a very useful property of softmax which helps the training process.
</div>


Now, because the model produces the outputs one at a time, we can assume that the model is selecting the word with the highest probability from that probability distribution and throwing away the rest. That's one way to do it (called greedy decoding). Another way to do it would be to hold on to, say, the top two words (say, 'I' and 'a' for example), then in the next step, run the model twice: once assuming the first output position was the word 'I', and another time assuming the first output position was the word 'a', and whichever version produced less error considering both positions #1 and #2 is kept. We repeat this for positions #2 and #3...etc. This method is called "beam search", where in our example, beam_size was two (meaning that at all times, two partial hypotheses (unfinished translations) are kept in memory), and top_beams is also two (meaning we'll return two translations). These are both hyperparameters that you can experiment with.



## Go Forth And Transform

I hope you've found this a useful place to start to break the ice with the major concepts of the Transformer. If you want to go deeper, I'd suggest these next steps:

* Read the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, the Transformer blog post ([Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)), and the [Tensor2Tensor announcement](https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html).
* Watch [Łukasz Kaiser's talk](https://www.youtube.com/watch?v=rBCqOTEfxvg) walking through the model and its details
* Play with the [Jupyter Notebook provided as part of the Tensor2Tensor repo](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)
* Explore the [Tensor2Tensor repo](https://github.com/tensorflow/tensor2tensor).

Follow-up works:

* [Depthwise Separable Convolutions for Neural Machine Translation](https://arxiv.org/abs/1706.03059)
* [One Model To Learn Them All](https://arxiv.org/abs/1706.05137)
* [Discrete Autoencoders for Sequence Models](https://arxiv.org/abs/1801.09797)
* [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198)
* [Image Transformer](https://arxiv.org/abs/1802.05751)
* [Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247)
* [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
* [Fast Decoding in Sequence Models using Discrete Latent Variables](https://arxiv.org/abs/1803.03382)
* [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

## Acknowledgements
Thanks to <a href="https://twitter.com/ilblackdragon">Illia Polosukhin</a>, <a href="http://jakob.uszkoreit.net/">Jakob Uszkoreit</a>, <a href="https://www.linkedin.com/in/llion-jones-9ab3064b">Llion Jones </a>, <a href="https://ai.google/research/people/LukaszKaiser">Lukasz Kaiser</a>, <a href="https://twitter.com/nikiparmar09">Niki Parmar</a>, and <a href="https://dblp.org/pers/hd/s/Shazeer:Noam">Noam Shazeer</a> for providing feedback on earlier versions of this post.

Please hit me up on <a href="https://twitter.com/JayAlammar">Twitter</a> for any corrections or feedback.
