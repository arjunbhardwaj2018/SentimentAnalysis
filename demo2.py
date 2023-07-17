import numpy as np
import pandas as pd

#import data
dataset = pd.read_csv(r"C:\Users\bhard\Downloads\drive-download-20230404T064236Z-001\a2_RestaurantReviews_FreshDump.tsv", delimiter = '\t', quoting = 3)
l=len(dataset)
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


for i in range(0, l):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

print(corpus)

#data transformation
# Loading BoW dictionary
from sklearn.feature_extraction.text import CountVectorizer
import pickle
cvFile=r"C:\Users\bhard\Downloads\drive-download-20230404T064236Z-001\c1_BoW_Sentiment_Model.pkl"
cv = pickle.load(open(cvFile, "rb"))

X_fresh = cv.transform(corpus).toarray()
print(X_fresh.shape)

#Predictions (via sentiment classifier)
import joblib
classifier = joblib.load(r"C:\Users\bhard\Downloads\drive-download-20230404T064236Z-001\c2_Classifier_Sentiment_Model")

y_pred = classifier.predict(X_fresh)
print(y_pred)

dataset['predicted_label'] = y_pred.tolist()
print(dataset.head())

#sasme it ok
dataset.to_csv(r"C:\Users\bhard\Downloads\drive-download-20230404T064236Z-001\c3_Predicted_Sentiments_Fresh_Dump.tsv", sep='\t', encoding='UTF-8', index=False)

#print pichart
count1 = (dataset['predicted_label'] == 1).sum()
count0 = (dataset['predicted_label'] == 0).sum()

# Import libraries
from matplotlib import pyplot as plt
# Creating dataset
Sentiments = ['Good Comments', 'Bad Comments']

data = [count1,count0]

# Creating plot
plt.pie(data, labels=Sentiments,autopct='%1.1f%%', startangle=90)

# show plot
plt.show()
