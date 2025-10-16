# Sentiment Analysis on Twitter Data

*A Progressive Exploration from TF-IDF to BERT Fine-Tuning*

##  Overview

This repository presents a series of sentiment analysis projects on an English-language Twitter dataset.
The goal is to explore how model performance evolves when moving from classical machine learning approaches to modern deep learning and transformer-based architectures.

Each approach is implemented in a separate folder:

1. **TF-IDF**
2. **Word2Vec**
3. **BERT & DistilBERT**

---

## 1. TF-IDF

The first model uses traditional NLP techniques to classify tweets as **positive** or **negative**.

**Key features**

* Feature extraction using [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
* Classification with [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* Lightweight and interpretable baseline for comparison

**Dataset structure**

| Column  | Description                            |
| :------ | :------------------------------------- |
| `ID`    | Unique identifier for each tweet       |
| `Text`  | Tweet content                          |
| `Label` | Sentiment (1 = positive, 0 = negative) |

---

## 2. Word2Vec

Building upon the baseline, this approach uses **word embeddings** and a **neural network** for richer semantic understanding.

**Key features**

* Implemented with [PyTorch](https://pytorch.org/)
* Pretrained **Word2Vec** embeddings as input features
* Fully connected neural network trained on the same dataset
* Demonstrates the advantage of distributed representations over TF-IDF

---

## 3. BERT & DistilBERT

The final stage applies **transfer learning** by fine-tuning powerful pretrained language models.

**Models used**

* [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
* [DistilBERT-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)

**Highlights**

* Fine-tuning on the Twitter sentiment dataset using [PyTorch](https://pytorch.org/)
* Comparison of performance between full BERT and the lighter DistilBERT
* Showcases how transformer models capture deep contextual meaning in text

---

## 📁 Repository Structure

```
├── input/                 # Dataset (train/test/validation)
├── TF-IDF/                # TF-IDF + Logistic Regression implementation
├── Word2Vec/              # DNN with Word2Vec embeddings
├── BERT - DistilBERT/     # BERT & DistilBERT fine-tuning
└── README.md
```

---

## ✨ Author

**Dimitris Andreakis**

University of Athens — Department of Informatics and Telecommunications
