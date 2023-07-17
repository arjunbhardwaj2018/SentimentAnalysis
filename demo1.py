import numpy as np
import pandas as pd

#import data
dataset = pd.read_csv(r"C:\Users\bhard\Downloads\drive-download-20230404T064236Z-001\a1_RestaurantReviews_HistoricDump.tsv", delimiter = '\t', quoting = 3)
print(dataset.shape)
print(dataset.head())

#data cleaning
import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

#filteration start
corpus=[]

for i in range(0, 900):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

print(corpus)

#data transformation
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1420)

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Saving BoW dictionary to later use in prediction
import pickle
bow_path = r"C:\Users\bhard\Downloads\drive-download-20230404T064236Z-001\c1_BoW_Sentiment_Model.pkl"
pickle.dump(cv, open(bow_path, "wb"))

#Dividing dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Model fitting (Naive Bayes)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Exporting NB Classifier to later use in prediction
import joblib
joblib.dump(classifier, r"C:\Users\bhard\Downloads\drive-download-20230404T064236Z-001\c2_Classifier_Sentiment_Model")


#model performance
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))