# Fake-News-Detector

## AIM:

The aim of this project is to detect the Fake news.

## PROCEDURE:
```
1.Inbuilt all the libraries which is involved in the algorithm to detect the fake news such as pandas which is used for analysis and manipulating.

2.Numpy is used for numerical computing.

3. matplotlib.pyplot  gives the data visualization as the output of the solution.
   
4.seaborn  provides a high-level interface for creating attractive and informative statistical graphics.

5.sklearn.model_selection  is a model selection which is used for evaluation in machine learning.

6. sklearn.feature_extraction.text is designed for converting text data into numerical feature vectors.

7.sklearn.linear_model is used for is used as a linear modeling techniques in machine learning such as Linear Regression,Logistic Regression,Ridge and Lasso Regression and etc.

8.sklearn.metric is  used for evaluating the performance of machine learning models such as Accuracy,Precision, Recall, F1-Score,Confusion Matrix,ROC Curve and AUC and Classification Report.

```

## ALGORITHM:
```

1.Load the dataset.

2. Label the data.
   
3.Concatenate the two datasets.

4.Shuffle the dataset.

5. Function to clean text.
   
6.Clean the text data.

7.Split the dataset and Train-Test Split.

8.Vectorize using TF-IDF.

9. Initialize and train the model and predict on the test set.
    
10.Evaluate the model and plot confusion matrix.
```

## PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df_fake = pd.read_csv('Fake.csv')
df_real = pd.read_csv('True.csv')

# Label the data
df_fake['label'] = 0
df_real['label'] = 1

# Concatenate the two datasets
df = pd.concat([df_fake, df_real], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

import re

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Clean the text data
df['text'] = df['text'].apply(clean_text)

# Split the dataset
X = df['text']
y = df['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
```
## OUTPUT:

![Screenshot 2024-10-19 233250](https://github.com/user-attachments/assets/ab79c2ec-220a-4a18-8054-e0b1582d2970)

## RESULT:

Thus the  model is tested on  accuracy, precision, recall, and F1-score.Real-Time using the given dataset.

