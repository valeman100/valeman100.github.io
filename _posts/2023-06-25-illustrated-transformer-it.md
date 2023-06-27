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

## Ora stiamo codificando!

Come abbiamo già accennato, l'encoder riceve una lista di vettori in input. Elabora questa lista passandoli attraverso uno strato di 'self-attention', per poi processarli attraverso una rete neurale feed-forward la quale genera l'output per il codificatore successivo.


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/encoder_with_tensors_2.png" />
  <br />
  Ogni parola passa attraverso un processo di self-attention. Per poi attraversare una rete neurale feed-forward - la stessa rete con ogni vettore che scorre attraverso di essa separatamente.
</div>

## Self-Attention in generale
Non lasciarti ingannare dal fatto che io utilizzi la parola "self-attention" come se fosse un concetto che tutti dovrebbero conoscere. Personalmente, non ero mai venuto a conoscenza di questo concetto prima di leggere l'articolo "Attention is All You Need". Vediamo come funziona.

Supponiamo che la seguente frase sia una frase di input che vogliamo tradurre:

"```The animal didn't cross the street because it was too tired```"

A cosa si riferisce "it" in questa frase? Si riferisce alla strada o all'animale? È una domanda semplice per un essere umano, ma non altrettanto semplice per un algoritmo.

Quando il modello elabora la parola "it", la self-attention gli consente di associare "it" a "animal".

Man mano che il modello elabora ogni parola (ogni posizione nella sequenza di input), il self-attention gli consente di guardare altre posizioni nella sequenza di input per trovare indizi che possano aiutare a ottenere una migliore codifica per questa parola.

Se hai familiarità con le reti neurali ricorrenti (RNN), pensa a come mantenere uno stato nascosto consente a un'RNN di incorporare la sua rappresentazione delle parole/vettori precedenti che ha elaborato con quella corrente che sta elaborando. Il self-attention è il metodo che il Transformer utilizza per "incorporare" la "comprensione" di altre parole pertinenti in quella che stà elaborando attualmente.


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self-attention_visualization.png" />
  <br />
  Mentre stiamo codificando la parola "it" nel codificatore n. 5 (il codificatore superiore nello stack), parte del meccanismo di attenzione si stava concentrando su "The animal" e ha incorporato una parte della sua rappresentazione nella codifica di "it".
</div>


Assicurati di dare un'occhiata al [notebook Tensor2Tensor](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) in cui puoi caricare un modello Transformer e esaminarlo utilizzando questa visualizzazione interattiva.



## Self-Attention in Dettaglio
Iniziamo guardando come calcolare la self-attention utilizzando vettori e successivamente vedremo come viene effettivamente implementata -- utilizzando le matrici.

**Primo passo** nel calcolare la self-attention è creare tre vettori da ciascun vettore di input del codificatore (in questo caso, l'embedding di ogni parola). Quindi, per ogni parola, creiamo un vettore di Query, un vettore di Key e un vettore di Value. Questi vettori vengono creati moltiplicando l'embedding per tre matrici che abbiamo addestrato durante il processo di training.

Nota che questi nuovi vettori sono di dimensioni più piccole rispetto al vettore di embedding. La loro dimensionalità è di 64, mentre l'embedding e i vettori di input/output del codificatore hanno una dimensionalità di 512. Non è obbligatorio che siano più piccoli, è una scelta architetturale per rendere il calcolo della multiheaded attention (in gran parte) costante.


<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self_attention_vectors.png" />
  <br />
  Moltiplicando <span class="encoder">x1</span> per la matrice dei pesi del <span class="decoder">WQ</span> otteniamo <span class="decoder">q1</span>, il vettore "query" associato con quella parola. Alla fine creiamo una proiezione "query", "key" e "value" per ogni parola nella frase di input.
</div>

<br />
<br />

Cosa sono i vettori "query", "key" e "value"?
<br />
<br />
Sono astrazioni che sono utili per calcolare e comprendere l'attenzione. Una volta che procedi nella lettura di come viene calcolata l'attenzione di seguito, saprai praticamente tutto ciò che devi sapere sul ruolo che ciascuno di questi vettori svolge.

**Secondo passo** nel calcolare la self-attention è calcolare uno score. Supponiamo di calcolare la self-attention per la prima parola in questo esempio, "Thinking". Dobbiamo valutare ogni parola della frase di input rispetto a questa parola. Lo score determina quanto focalizzare altre parti della frase di input mentre codifichiamo una parola in una determinata posizione.

Lo score viene calcolato prendendo il prodotto scalare del <span class="decoder">vettore di query</span> con il <span class="context">vettore di key</span> della rispettiva parola che stiamo valutando. Quindi, se stiamo elaborando la self-attention per la parola in posizione <span class="encoder">#1</span>, il primo score sarebbe il prodotto scalare tra <span class="decoder">q1</span> e <span class="context">k1</span>. Il secondo score sarebbe il prodotto scalare tra <span class="decoder">q1</span> e <span class="context">k2</span>.


<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self_attention_score.png" />
  <br />

</div>

<br />


Il **terzo e quarto passaggio** consistono nel dividere i punteggi per 8 (la radice quadrata della dimensione dei vettori chiave utilizzati nell'articolo, ossia 64). Ciò porta a ottenere gradienti più stabili. Potrebbero esserci altri valori possibili qui, ma questo è il valore predefinito. Successivamente, il risultato viene sottoposto a un'operazione softmax. La softmax normalizza i punteggi in modo che siano tutti positivi e la loro somma sia uguale a 1.


<br />


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/self-attention_softmax.png" />
  <br />

</div>

Questo punteggio softmax determina quanto ogni parola sarà espressa in questa posizione. Chiaramente, la parola in questa posizione avrà il punteggio softmax più alto, ma talvolta è utile concentrarsi su un'altra parola rilevante per la parola corrente.

<br />


Il **quinto passaggio** consiste nel moltiplicare ciascun vettore di value per il punteggio softmax (in preparazione per la loro somma). L'intuizione qui è mantenere integri i valori della/e parola/e su cui vogliamo concentrarci e "affogare" le parole irrilevanti (moltiplicandole per numeri piccoli come 0,001, ad esempio).

Il **sesto passaggio** serve a sommare i vettori di value precedentemente pesati. Questo produce l'output del layer di self-attention per questa posizione (ovvero per la prima parola).

<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/self-attention-output.png" />
  <br />
</div>

Questo conclude il calcolo dell'self-attention. Il vettore risultante è quello che possiamo inviare alla rete neurale a feed-forward. Nell'implementazione effettiva, tuttavia, questo calcolo viene effettuato in forma matriciale per una elaborazione più veloce. Quindi vediamo ora questo aspetto, ora che abbiamo compreso l'intuizione del calcolo a livello di singola parola.


## Calcolo Matriciale per la Self-Attention
**Il primo passo** consiste nel calcolare le matrici di Query, Key e Value. Lo facciamo raggruppando le nostre rappresentazioni in una matrice <span class="encoder">X</span> e moltiplicandola per le matrici di pesi che abbiamo allenato (<span class="decoder">WQ</span>, <span class="context">WK</span>, <span class="step_no">WV</span>).

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/self-attention-matrix-calculation.png" />
  <br />
    Ogni riga nella matrice <span class="encoder">X</span> corrisponde a una parola nella frase di input. Possiamo notare la differenza di dimensione tra il vettore di embedding (512, o 4 caselle nella figura) e i vettori q/k/v (64, o 3 caselle nella figura).
</div>

<br />

**Infine**, dato che stiamo lavorando con matrici, possiamo condensare i passaggi dal due al sei in una formula per calcolare gli output del layer di self-attention.

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/self-attention-matrix-calculation-2.png" />
  <br />
  Il calcolo della self-attention in forma matriciale.
</div>

<br />


<br />

## La bestia con molte teste

Il paper ha ulteriormente migliorato il layer di auto-attenzione aggiungendo un meccanismo chiamato "multi-headed" attention. Questo migliora le prestazioni del layer di attenzione in due modi:

1. Espande la capacità del modello di concentrarsi su diverse posizioni. Sì, nell'esempio precedente, z1 contiene un po' di ogni altra codifica, ma potrebbe essere dominato dalla parola stessa. Se stiamo traducendo una frase come "The animal didn't cross the street because it was too tired", sarebbe utile sapere a quale parola si riferisce "it".

2. Fornisce al layer di attenzione più "sottospazi di rappresentazione". Come vedremo in seguito, con la multi-headed attention non abbiamo solo un set di matrici di pesi Query/Key/Value (il Transformer utilizza otto attenzioni distinte, quindi otteniamo otto set per ogni encoder/decoder). Ogni set di matrici viene inizializzato casualmente. Successivamente, dopo l'addestramento, ogni set viene utilizzato per proiettare gli embedding di input (o vettori da encoder/decoder inferiori) in un diverso sottospazio di rappresentazione.


 <div class="img-div-any-width" markdown="0">
   <img src="/images/t/transformer_attention_heads_qkv.png" />
   <br />
   Con la multi-headed attention, manteniamo separate matrici di pesi Q/K/V per ogni attenzione, ottenendo diverse matrici Q/K/V. Come abbiamo fatto prima, moltiplichiamo X per le matrici WQ/WK/WV per ottenere le matrici Q/K/V.
 </div>

 <br />
Se eseguiamo il calcolo di auto-attenzione descritto sopra, ma otto volte diverse con diverse matrici di pesi, otteniamo otto diverse matrici Z.

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_attention_heads_z.png" />
  <br />

</div>

 <br />

Questo ci pone di fronte a una sfida. Il livello di feed-forward non si aspetta otto matrici, ma si aspetta una singola matrice (un vettore per ogni parola). Quindi abbiamo bisogno di un modo per condensare queste otto matrici in una singola matrice.

Come facciamo? Concateniamo le matrici, quindi le moltiplichiamo per una matrice di pesi aggiuntiva WO.


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_attention_heads_weight_matrix_o.png" />
  <br />

</div>

Praticamente è tutto quello che c'è da sapere sulla multi-headed self-attention. Mi rendo conto che ci sono parecchie matrici. Cercherò di metterle tutte in un'unica immagine in modo da poterle vedere tutte insieme.

<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_multi-headed_self-attention-recap.png" />
  <br />

</div>

<br />

Ora che abbiamo parlato delle attention heads, torniamo all'esempio precedente per vedere su cosa si stanno concentrando le diverse attenzioni mentre codifichiamo la parola "it" nella nostra frase di esempio:

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self-attention_visualization_2.png" />
  <br />
  Mentre codifichiamo la parola "it", una delle attenzioni si concentra principalmente su "the animal", mentre un'altra si concentra su "tired" -- in un certo senso, la rappresentazione della parola "it" nel modello include un po' della rappresentazione sia di "animal" che di "tired".
</div>

<br />

Tuttavia, se aggiungiamo tutte le attenzioni all'immagine, può essere più difficile interpretarle:
<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self-attention_visualization_3.png" />
  <br />
</div>



## Rappresentare l'Ordine della Sequenza Utilizzando l'Encoding Posizionale
Una cosa che manca nel modello, come lo abbiamo descritto finora, è un modo per tenere conto dell'ordine delle parole nella sequenza di input.

Per affrontare questo problema, il transformer aggiunge un vettore a ciascun embedding di input. Questi vettori seguono un pattern specifico che il modello impara, il quale aiuta a determinare la posizione di ogni parola, o la distanza tra diverse parole nella sequenza. L'intuizione qui è che aggiungere questi valori agli embeddings fornisce distanze significative tra i vettori di embedding una volta che sono proiettati nei vettori Q/K/V e durante la dot-product attention.

<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_positional_encoding_vectors.png" />
  <br />
  Per dare al modello un senso dell'ordine delle parole, aggiungiamo vettori di encoding posizionale, i cui valori seguono un pattern specifico.
</div>
  <br />


Se assumiamo che l'embedding abbia una dimensionalità di 4, gli encoding posizionali effettivi sarebbero simili a questo:

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_positional_encoding_example.png" />
  <br />
 Un esempio reale di encoding posizionale con una dimensione di embedding esemplificativa di 4.
</div>

  <br />

A che cosa potrebbe assomigliare questo pattern?

Nella figura seguente, ogni riga corrisponde a un encoding posizionale di un vettore. Quindi, la prima riga sarebbe il vettore che aggiungeremmo all'embedding della prima parola in una sequenza di input. Ogni riga contiene 512 valori, ognuno con un valore compreso tra 1 e -1. Li abbiamo colorati per rendere visibile il pattern.



<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_positional_encoding_large_example.png" />
  <br />
  Un esempio reale di encoding posizionale per 20 parole (righe) con una dimensione di embedding di 512 (colonne). Puoi vedere che sembra diviso a metà lungo il centro. Questo perché i valori della metà sinistra sono generati da una funzione (che utilizza il seno), mentre la metà destra è generata da un'altra funzione (che utilizza il coseno). Vengono quindi concatenati per formare ciascuno dei vettori di encoding posizionale.
</div>


La formula per la codifica posizionale è descritta nell'articolo (sezione 3.5). Puoi vedere il codice per generare le codifiche posizionali in [```get_timing_signal_1d()```](https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py). Questo non è l'unico metodo possibile per la codifica posizionale. Tuttavia, offre il vantaggio di poter scalare a lunghezze di sequenze non viste in precedenza (ad esempio, se al nostro modello addestrato viene chiesto di tradurre una frase più lunga rispetto a quelle presenti nel nostro set di addestramento).

**Aggiornamento luglio 2020:** 
La codifica posizionale mostrata sopra proviene dall'implementazione Tranformer2Transformer del Transformer. Il metodo mostrato nell'articolo è leggermente diverso in quanto non concatena direttamente, ma intreccia i due segnali. La figura seguente mostra come appare. [Qui il codice che lo genera:](https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb):

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/attention-is-all-you-need-positional-encoding.png" />
  <br />
</div>

## I Residuals
Un dettaglio dell'architettura dell'encoder che dobbiamo menzionare prima di procedere è che ogni sottolivello (self-attention, ffnn) in ogni encoder ha una connessione residua attorno ad esso ed è seguito da un passaggio di [layer-normalization](https://arxiv.org/abs/1607.06450).

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_resideual_layer_norm.png" />
  <br />
</div>

Se dovessimo visualizzare i vettori e l'operazione di normalizzazione del layer associata alla self-attention, sarebbe così:

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_resideual_layer_norm_2.png" />
  <br />
</div>

Questo vale anche per i sottolivelli del decoder. Se pensassimo a un Transformer con 2 encoder e decoder che si susseguono, sarebbe qualcosa del genere:

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_resideual_layer_norm_3.png" />
  <br />
</div>


## La parte del Decoder
Ora che abbiamo coperto la maggior parte dei concetti relativi al lato encoder, conosciamo fondamentalmente il funzionamento dei componenti del decoder. Ma diamo un'occhiata a come lavorano insieme.

L'encoder inizia elaborando la sequenza di input. L'output dell'encoder superiore viene quindi trasformato in un insieme di vettori di attention K e V. Questi vettori verranno utilizzati da ciascun decoder nel suo strato di "encoder-decoder attention", che aiuta il decoder a concentrarsi sui punti appropriati nella sequenza di input:


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_decoding_1.gif" />
  <br />
  Dopo aver completato la fase di codifica, iniziamo la fase di decodifica. Ogni passaggio nella fase di decodifica produce un elemento dalla sequenza di output (la frase tradotta in inglese in questo caso).
</div>

I passaggi successivi ripetono il processo fino a quando non viene raggiunto un simbolo speciale di fine frase (<end of sentence>) che indica che il decoder del transformer ha completato la sua generazione di output. L'output di ogni passaggio viene alimentato al decoder inferiore nel passaggio successivo, e i decoder trasmettono i loro risultati di decodifica proprio come hanno fatto gli encoder. E proprio come abbiamo fatto con gli input dell'encoder, incorporiamo e aggiungiamo una codifica di posizione a questi input del decoder per indicare la posizione di ciascuna parola.

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_decoding_2.gif" />
  <br />

</div>

Gli strati di self-attention nel decoder operano in modo leggermente diverso rispetto a quelli nell'encoder:

Nel decoder, lo strato di self-attention può considerare solo le posizioni precedenti nella sequenza di output. Ciò viene fatto mascherando le posizioni future (impostandole a ```-inf```) prima del passaggio softmax nel calcolo della self-attention.

Lo strato di "Encoder-Decoder Attention" funziona proprio come la multiheaded self-attention, ad eccezione che crea la matrice delle Queries dallo strato sottostante e prende le matrici delle Keys e delle Values dall'output dello stack dell'encoder.

## Lineare finale e il layer di Softmax.

La pila del decoder produce un vettore di numeri decimali. Come lo convertiamo in una parola? Questo è il compito del livello lineare finale, seguito da un livello Softmax.

Il livello lineare è un semplice neurone completamente connesso che proietta il vettore prodotto dalla pila dei decoder in un vettore molto, molto più ampio chiamato vettore dei logit.

Supponiamo che il nostro modello conosca 10.000 parole inglesi uniche (il "vocabolario di output" del nostro modello) apprese dal set di dati di addestramento. Questo renderebbe il vettore dei logit largo 10.000 celle, ognuna corrispondente al punteggio di una parola unica. Così interpretiamo l'output del modello seguito dal livello lineare.

Il livello Softmax trasforma quindi questi punteggi in probabilità (tutte positive, che sommano a 1,0). Viene selezionata la cella con la probabilità più alta e la parola associata ad essa viene prodotta come output per questo passo temporale.

  <br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_decoder_output_softmax.png" />
  <br />
  Questa figura parte dal basso con il vettore prodotto come output della pila dei decoder. Viene quindi convertito in una parola di output.
</div>

  <br />

## Riepilogo del Training
Ora che abbiamo coperto l'intero processo di passaggio in avanti attraverso un Transformer addestrato, sarebbe utile dare un'occhiata all'intuizione dell'addestramento del modello.

Durante l'addestramento, un modello non addestrato passerebbe attraverso lo stesso passaggio in avanti. Ma poiché lo stiamo addestrando su un set di dati di addestramento etichettato, possiamo confrontare il suo output con l'output corretto effettivo.

Per visualizzare questo, supponiamo che il nostro vocabolario di output contenga solo sei parole ("a", "am", "i", "thanks", "student", and "\<eos\>" (abbreviato per 'end of sentence')).

 <div class="img-div" markdown="0">
   <img src="/images/t/vocabulary.png" />
   <br />
   Il vocabolario di output del nostro modello viene creato nella fase di preelaborazione prima di iniziare l'addestramento.
 </div>

Una volta definito il nostro vocabolario di output, possiamo utilizzare un vettore della stessa larghezza per indicare ogni parola nel nostro vocabolario. Questo è anche noto come codifica one-hot. Ad esempio, possiamo indicare la parola "am" utilizzando il seguente vettore:

<div class="img-div" markdown="0">
  <img src="/images/t/one-hot-vocabulary-example.png" />
  <br />
  Esempio: codifica one-hot del nostro vocabolario di output
</div>

Dopo questo riepilogo, discutiamo la funzione di perdita del modello -- la metrica che stiamo ottimizzando durante la fase di addestramento per portare a un modello addestrato e, si spera, sorprendentemente accurato.

## Loss Function
Supponiamo di essere nella fase di addestramento del nostro modello. Supponiamo che sia il nostro primo passo nella fase di addestramento e che lo stiamo addestrando su un semplice esempio -- tradurre "merci" in "thanks".

Ciò significa che vogliamo che l'output sia una distribuzione di probabilità che indichi la parola "grazie". Ma poiché questo modello non è ancora addestrato, ciò è improbabile che accada fin da subito.

<div class="img-div" markdown="0">
  <img src="/images/t/transformer_logits_output_and_label.png" />
  <br />
  Poiché i parametri (pesi) del modello sono inizializzati casualmente, il modello (non addestrato) produce una distribuzione di probabilità con valori arbitrari per ogni cella/parola. Possiamo confrontarla con l'output effettivo, quindi regolare tutti i pesi del modello utilizzando la retropropagazione per avvicinare l'output all'output desiderato.
</div>

<br />

Come confronti due distribuzioni di probabilità? Semplicemente sottraendo una dall'altra. Per ulteriori dettagli, guarda [cross-entropy](https://colah.github.io/posts/2015-09-Visual-Information/) e [Kullback–Leibler divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained).

Ma nota che questo è un esempio semplificato. In modo più realistico, useremo una frase più lunga di una parola. Ad esempio -- input: "je suis étudiant" e output previsto: "i am a student". Ciò significa che vogliamo che il nostro modello produca successivamente distribuzioni di probabilità in cui:

 * Ogni distribuzione di probabilità è rappresentata da un vettore di larghezza vocab_size (6 nel nostro esempio di esempio, ma più realisticamente un numero come 30.000 o 50.000)
 * La prima distribuzione di probabilità ha la probabilità più alta nella cella associata alla parola "i"
 * La seconda distribuzione di probabilità ha la probabilità più alta nella cella associata alla parola "am"
 * E così via, fino a quando la quinta distribuzione di output indica il simbolo '```<end of sentence>```' che ha anch'esso una cella associata dall'elemento di vocabolario di 10.000 elementi.


 <div class="img-div" markdown="0">
   <img src="/images/t/output_target_probability_distributions.png" />
   <br />
   Le distribuzioni di probabilità di destinazione su cui addestreremo il nostro modello nell'esempio di addestramento per una frase campione.
 </div>

<br />

Dopo aver addestrato il modello per un tempo sufficiente su un dataset abbastanza ampio, ci aspetteremmo che le distribuzioni di probabilità prodotte assomiglino a queste:

  <div class="img-div" markdown="0">
    <img src="/images/t/output_trained_model_probability_distributions.png" />
    <br />
    Speriamo che, dopo l'addestramento, il modello produca la traduzione corretta che ci aspettiamo. Naturalmente, questo non è un vero indicatore se questa frase faceva parte del dataset di addestramento (vedi: <a href="https://www.youtube.com/watch?v=TIgfjmp-4BA">validazione incrociata</a>). Notate che ogni posizione ottiene un po' di probabilità anche se è improbabile che sia l'output di quel passo temporale: questa è una proprietà molto utile della softmax che aiuta il processo di addestramento.
</div>


Ora, poiché il modello produce le uscite una alla volta, possiamo assumere che il modello selezioni la parola con la probabilità più alta da quella distribuzione di probabilità e scarti il resto. Questo è un modo per farlo (chiamato decodifica greedy). Un altro modo per farlo sarebbe quello di mantenere, ad esempio, le prime due parole (ad esempio, 'I' e 'a'), quindi nel passaggio successivo, eseguire il modello due volte: una volta assumendo che la prima posizione di output fosse la parola 'Io' e un'altra volta assumendo che la prima posizione di output fosse la parola 'un', e la versione che produce meno errore considerando entrambe le posizioni #1 e #2 viene conservata. Ripetiamo questo per le posizioni #2 e #3...ecc. Questo metodo si chiama "beam search", dove nel nostro esempio, beam_size era due (il che significa che in ogni momento, due ipotesi parziali (traduzioni incomplete) vengono conservate in memoria), e top_beams è anche due (il che significa che restituiremo due traduzioni). Questi sono entrambi iperparametri con cui puoi sperimentare.



## Avanti e trasforma

Spero che tu abbia trovato questo un punto di partenza utile per rompere il ghiaccio con i concetti principali del Transformer. Se vuoi approfondire, ti suggerisco i seguenti passaggi successivi:

* Leggi l'articolo [Attention Is All You Need](https://arxiv.org/abs/1706.03762), il blogpost sul transformer ([Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)), e il [Tensor2Tensor announcement](https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html).
* Guarda [Łukasz Kaiser's talk](https://www.youtube.com/watch?v=rBCqOTEfxvg) che illustra il modello e i suoi dettagli.
* Sperimenta con [Jupyter Notebook provided as part of the Tensor2Tensor repo](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)
* Esplora [Tensor2Tensor repo](https://github.com/tensorflow/tensor2tensor).

Lavori correlati:

* [Depthwise Separable Convolutions for Neural Machine Translation](https://arxiv.org/abs/1706.03059)
* [One Model To Learn Them All](https://arxiv.org/abs/1706.05137)
* [Discrete Autoencoders for Sequence Models](https://arxiv.org/abs/1801.09797)
* [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198)
* [Image Transformer](https://arxiv.org/abs/1802.05751)
* [Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247)
* [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
* [Fast Decoding in Sequence Models using Discrete Latent Variables](https://arxiv.org/abs/1803.03382)
* [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

## Riconoscimenti
Grazie a <a href="https://twitter.com/ilblackdragon">Illia Polosukhin</a>, <a href="http://jakob.uszkoreit.net/">Jakob Uszkoreit</a>, <a href="https://www.linkedin.com/in/llion-jones-9ab3064b">Llion Jones </a>, <a href="https://ai.google/research/people/LukaszKaiser">Lukasz Kaiser</a>, <a href="https://twitter.com/nikiparmar09">Niki Parmar</a>, and <a href="https://dblp.org/pers/hd/s/Shazeer:Noam">Noam Shazeer</a> per aver fornito feedback sulle versioni precedenti di questo post.

Per favore contatta Jay Alammar su <a href="https://twitter.com/JayAlammar">Twitter</a> per qualsiasi correzione o feedback sull'articolo originario.

Altrimenti per correzione o feedback su questa traduzione contattate Valerio Mannucci su <a href="https://twitter.com/Valeman100">Twitter</a>.
