
# <concepts guide>

> This section outlines the key concepts and theory behind the project, providing the necessary background to understand and interpret the time series analysis and forecasting techniques used.

## What is NLP?

Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It enables machines to read, interpret, understand, and generate human language in a meaningful way. In this project, NLP is used to automatically categorize credit card transactions based on their text descriptions, making it easier to analyze personal spending patterns.

## TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) is a numerical statistic that reflects how important a word is to a document in a collection or corpus. It combines two metrics:

- Term Frequency (TF): Measures how frequently a term appears in a document.
- Inverse Document Frequency (IDF): Penalizes common terms that appear in many documents and boosts rare, more informative words.
  
In this project, TF-IDF transforms transaction descriptions into numerical vectors, allowing machine learning models to identify patterns based on word importance.

## BERT

BERT (Bidirectional Encoder Representations from Transformers) is a powerful pre-trained language model developed by Google. It captures deep contextual meaning by reading text in both directions (left-to-right and right-to-left). Unlike traditional vectorization methods, BERT understands the nuance and context of words in a sentence.

In this project, BERT embeddings help improve the classification accuracy by providing richer semantic representations of transaction descriptions.

## Logistic Regression 

Logistic Regression is a supervised machine learning algorithm used for classification tasks. It models the probability that a given input belongs to a particular category. It outputs values between 0 and 1 using the sigmoid function, and classifies inputs based on a threshold.

In the context of this project, logistic regression is used to classify transaction descriptions into predefined categories (e.g., groceries, transportation, dining) based on their TF-IDF or BERT-transformed features.
