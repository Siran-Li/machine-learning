# Complete Sentence Detection for Speech Recognition Systems

## Project 2 (EPFL Machine Learning Course CS-433)

This is a repository for all code of project 2

Members:

- Bohan Wang (321293)

- Ke Wang (326760)

- Siran Li (321825)

## Datasets 

1. News reports: 143, 000 articles from 15 American publications [[1]](#1).
2. Ted 2020 Parallel Sentences Corpus: around 4000 TED Talk transcripts from July 2020 [[2]](#2).
3. Wikipedia corpus: over 10 million topics [[3]](#3).
4. Topical-Chat: human dialog conversations spanning 8 broad topics [[4]](#4).

## Transformers

Note all below pre-trained transformers are from Hugging face [[5]](#5).

1. Generative Pre-trained Transformer 2 (GPT-2) [[6]](#6).
2. Bidirectional Encoder Representations from Trans-formers (BERT) [[7]](#7).
3. Big Bird: Transformers for Longer Sequences [[8]](#8).

## Notes
The packages used in the project can be installed using:

``pip install datasets``

``pip install transformers``

``pip install ntlk``

``pip install Sentencepiece``

## Structure
**models.py:** contains the model definition code of BiLSTM, TextCNN and Transformer class

**utils.ipynb:** contains the helper functions for pre-processing

**Pre-processing.ipynb:** contains the code to preprocess raw text 

**train_models.ipynb:** contains the code to: 
 - Fine-tune BERT on standard dataset
 - Fine-tune GPT2 on standard dataset
 - Fine-tune BIGBIRD on standard dataset
 - BERT word embedding + BiLSTM
 - BERT word embedding + TextCNN
 - Fine-tune BERT on large data set
 - Fine-tune BERT with multi-label data

**random_forest.ipynb:** contains the code to aggregate five trained models with random forest.

## Instuctions
The preprocessing of raw text can be reproduced in:

``Pre-processing.ipynb``

However, as the raw data is too large. We didn't put them on Github. You can directly use train and test the model using processed datasets (without running Pre-processing).

You can reproduce the test performances of different models in:

``reproduce.ipynb``

You can train the models in:

``train_models.ipynb``

All the necessary datasets and models for training and reproducing the results using ``reproduce.ipynb`` and ``train_models.ipynb`` can be downloaded at: 
 https://drive.google.com/drive/folders/1sRMolxsVHLiLphfnS4NpDM0766XFOkZU?usp=sharing

## References
<a id="1">[1]</a> 
A.   Thompson, (2017)
“All   the   news:   143,000   articles   from   15   americanpublications,”
https://www.kaggle.com/snapcrack/all-the-new

<a id="2">[2]</a> 
N.   Reimers   and   I.   Gurevych, (2020)
“Making   monolingual   sentence   em-beddings   multilingual   using   knowledge   distillation,” arXiv   preprintarXiv:2004.09813

<a id="3">[3]</a> 
W.   Foundation.   Wikimedia   downloads.
[Online].   Available:   https://dumps.wikimedia.org

<a id="4">[4]</a> 
K.  Gopalakrishnan,  B.  Hedayatnia,  Q.  Chen,  A.  Gottardi,  S.  Kwatra,A. Venkatesh, R. Gabriel, D. Hakkani-T ̈ur, and A. A. AI, (2019)
“Topical-chat:Towards  knowledge-grounded  open-domain  conversations.”  inINTER-SPEECH, pp. 1891–1895

<a id="5">[5]</a> 
Hugging face is an AI community to provide open source NLP softwares, https://huggingface.co/

<a id="6">[6]</a> 
A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskeveret al.,“Language  models  are  unsupervised  multitask  learners,”OpenAI  blog,vol. 1, no. 8, p. 9, 2019.

<a id="7">[7]</a> 
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-trainingof  deep  bidirectional  transformers  for  language  understanding,”arXivpreprint arXiv:1810.04805, 2018.

<a id="8">[8]</a> 
M.   Zaheer,   G.   Guruganesh,   K.   A.   Dubey,   J.   Ainslie,   C.   Alberti,S.  Ontanon,  P.  Pham,  A.  Ravula,  Q.  Wang,  L.  Yanget  al.,  “Big  bird:Transformers for longer sequences.” inNeurIPS, 2020.



