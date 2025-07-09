
# concepts guide
> This section outlines the key concepts and theory behind the project, providing the necessary background to understand and interpret the natural language processing, supervised learning and regression techniques used. 

## What is NLP?

Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It enables machines to read, interpret, understand, and generate human language in a meaningful way. In this project, NLP is used to automatically categorize credit card transactions based on their text descriptions, making it easier to analyze personal spending patterns.

## TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) is a numerical statistic that reflects how important a word is to a document in a collection or corpus. It combines two metrics:

- Term Frequency (TF): Measures how frequently a term appears in a document.
- Inverse Document Frequency (IDF): Penalizes common terms that appear in many documents and boosts rare, more informative words.
  
In this project, TF-IDF transforms transaction descriptions into numerical vectors, allowing machine learning models to identify patterns based on word importance.

1. Creating a vectorizer.

- TfidfVectorizer is a class from scikit-learn that transforms raw text into numerical feature vectors.
- It's used to reflect how important a word is to a document in a collection (in my case, collection is the NewDescription column)
- fit_transform() learns the vocabulary from the NewDescription column and transforms each text into a sparse numerical matrix (rows = samples, columns = features).
  - Output: X_tf is a sparse matrix of shape [number_of_samples, number_of_features].

``` python
from sklearn.feature_extraction.text import TfidfVectorizer

# TfidfVectorizer converts your text to numerical values.
vectorizer = TfidfVectorizer(ngram_range=(1,2), 
                             min_df=2,
                             sublinear_tf=True)
# sublinear_tf applies logarithmic scaling to term frequency. This helps reduce the influence of commonly repeated terms that might dominate otherwise.

X_tf = vectorizer.fit_transform(df['NewDescription'])
```

2. Splitting the vectors and their category labels into training set (80%) and testing set (20%).
   
- X_tf_train = TF-IDF vectors of 80% of the descriptions (training features).
- X_tf_test = TF-IDF vectors of 20% of the descriptions (test features).
- y_tf_train = Corresponding category labels (actual text) for the 80% training descriptions.
- y_tf_test = Corresponding category labels (actual text) for the 20% test descriptions.
   
4. Training the model on training set. Evaluating the model using the test set. 

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_tf_train, X_tf_test, y_tf_train, y_tf_test = train_test_split(X_tf, df['Category'], test_size=0.2)

model_tf = LogisticRegression()
#model = LogisticRegression(class_weight='balanced')

model_tf.fit(X_tf_train, y_tf_train)

from sklearn.metrics import classification_report, accuracy_score

y_tf_pred = model_tf.predict(X_tf_test)
print(classification_report(y_tf_test, y_tf_pred))

```

## BERT

BERT (Bidirectional Encoder Representations from Transformers) is a powerful pre-trained language model developed by Google. It captures deep contextual meaning by reading text in both directions (left-to-right and right-to-left). Unlike traditional vectorization methods, BERT understands the nuance and context of words in a sentence.

In this project, BERT embeddings help improve the classification accuracy by providing richer semantic representations of transaction descriptions.

## Logistic Regression 

Logistic Regression is a supervised machine learning algorithm used for classification tasks. It models the probability that a given input belongs to a particular category. It outputs values between 0 and 1 using the sigmoid function, and classifies inputs based on a threshold.

In the context of this project, logistic regression is used to classify transaction descriptions into predefined categories (e.g., groceries, transportation, dining) based on their TF-IDF or BERT-transformed features.
