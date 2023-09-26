---
layout: post
published: True
title: Il transformer illustrato - IT
---

## Disclaimer
Traduzione italiana di [The illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)\
Non sono un traduttore professionista. \
La proprietà intellettuale dell'articolo è di [Jay Alammar](http://jalammar.github.io/illustrated-transformer/)

Italian translation of [The illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)\
I'm not a professional translator. \
The intellectual property of the article is owned by [Jay Alammar](http://jalammar.github.io/illustrated-transformer/)

Nel [post precedente, abbiamo esaminato l'Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) -- un metodo onnipresente nei moderni modelli di deep learning. L'attention è uno strumento che ha contribuito a migliorare le prestazioni delle applicazioni di traduzione automatica che utilizzano modelli neurali. In questo post, esamineremo **Il Transformer**, un modello che utilizza l'attention per aumentare la velocità con cui queste reti possono essere addestrati. Il Trasformer ha perfino supera il modello di traduzione automatica neurale di Google in attività specifiche. Il più grande vantaggio, tuttavia, deriva dal modo in cui il Transformer si presta alla parallelizzazione. È infatti raccomandazione di Google Cloud sfruttare il Transformer come modello di riferimento per utilizzare la loro proposte di [Cloud TPU](https://cloud.google.com/tpu/). Proviamo a scomporre il modello e vediamo come funziona.

Il Transformer è stato proposto nell'articolo [Attention is All You Need](https://arxiv.org/abs/1706.03762). Una sua implementazione TensorFlow è disponibile come parte del pacchetto [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor). Il gruppo NLP di Harvard ha creato una [guida che spiega l'articolo con implementazioni in PyTorch](http://nlp.seas.harvard.edu/2018/04/03/attention.html). In questo post, cercheremo di semplificare un po' le cose e di introdurre i concetti uno per uno, sperando che sia più facile da capire per le persone senza una conoscenza approfondita dell'argomento.

**Aggiornamento 2020**: Ho creato il video "Transformer narrati" che è un approccio più soft all'argomento:

 <div style="text-align:center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/-QH8fRhqFHM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"  style="
 width: 100%;
 max-width: 560px;"
allowfullscreen></iframe>
</div>

<!--more-->

Read the full article on [Medium](https://medium.com/@val.mannucci/il-transformer-illustrato-it-37a78e3e2348)

Per favore contatta Jay Alammar su <a href="https://twitter.com/JayAlammar">Twitter</a> per qualsiasi correzione o feedback sull'articolo originario.

Altrimenti per correzione o feedback su questa traduzione contattate Valerio Mannucci su <a href="https://twitter.com/Valeman100">Twitter</a>.
![image](https://github.com/valeman100/valeman100.github.io/assets/57062687/db855665-2c96-4526-b81a-8048eef1525b)
