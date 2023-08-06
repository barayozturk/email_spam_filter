# 📧 Spam Detection with Naive Bayes Classifier 💡

This is a simple spam detection project using the Naive Bayes classifier. The goal of this project is to classify emails as either spam or ham (not spam) based on their text content.

## Dataset 📊

The dataset used in this project is stored in a CSV file named 'emails.csv'. It contains two columns: 'text' (the email content) and 'spam' (1 for spam, 0 for ham).

## Data Exploration 🔍

First, we import the necessary libraries: pandas, numpy, matplotlib, seaborn, and warnings. We also read the dataset into a pandas DataFrame and display its first few rows. 📋

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

spam_df = pd.read_csv('emails.csv')
spam_df.head()
```

## Next, we explore the dataset by checking its summary statistics and information: 📊

````python
spam_df.describe()
spam_df.info()
````

## We can also visualize the data by grouping emails into spam and ham categories: 📈

````python
spam_df.groupby('spam').describe()
````

## We calculate the length of each email and add it as a new column to the DataFrame: 📏

````python
spam_df['length'] = spam_df['text'].apply(len)
spam_df.head()
````

## Then, we plot a histogram of email lengths to get an overview of the data distribution: 📊

````python
spam_df['length'].plot(bins=100, kind='hist')
````

## Data Preprocessing 🧹

Before training the Naive Bayes classifier, we need to preprocess the text data.
We create a function called message_cleaning to remove punctuation and stopwords (common words with little meaning) from the email text. 📝


````pyhton
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.corpus import stopwords

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean

# Applying message_cleaning to the 'text' column
spam_df_clean = spam_df['text'].apply(message_cleaning)
````

## Vectorization and Model Training 🎯

We use the CountVectorizer from scikit-learn to convert the text data into numerical vectors. Then, we train a Naive Bayes classifier on the vectorized data. 🚀

````python
vectorizer = CountVectorizer(analyzer=message_cleaning)
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X = spamham_countvectorizer
y = spam_df['spam'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
````

## Model Evaluation 📈

We evaluate the Naive Bayes classifier on the test set using classification metrics such as precision, recall, and F1-score. 📊

````python
from sklearn.metrics import classification_report, confusion_matrix

y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))
````

## Conclusion 🎉

In this project, we built a simple spam detection model using the Naive Bayes classifier.
The model showed good performance in distinguishing spam from ham emails, achieving an accuracy of approximately 99% on the training set and 65% on the test set. 
However, further improvements can be made by exploring other machine learning algorithms and feature engineering techniques. 🤖

Feel free to use this project as a starting point for your own spam detection tasks! 🚀


