# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:14:15 2020

@author: suyog
"""

# Classification of Spam and Ham ( Non- Spam ) using Natural Language Processing ( NLP )

#Importing Libraries
import pandas as pd
import numpy as np

msg = pd.read_csv('SMSSpamCollection',sep='\t',names = ['Label','Message'])

#Feature Engineering

#Data Cleaning
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords #Stopwords will remove the general word of english like the,a,of,under etc
from nltk.stem.porter import PorterStemmer #Stemming process
from nltk.stem import WordNetLemmatizer #Lemmatization process

ps = PorterStemmer()
lm = WordNetLemmatizer()

corpus = [] #this will have our words and its frequency

#Data Cleaning and Pre processing
for i in range(0,len(msg)):
    review = re.sub('[^a-zA-Z]',' ',msg['Message'][i]) #Removing the commas and symbols
    review = review.lower() #change all words to lower case 
    review = review.split()
    
    #Now we use the stemming process to get the root word of the derived words
    #review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#You can use either Bag of Words aka CountVectorizer method or TFIDVectorizer method  
    
# Creating the Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()
word_freq = cv.vocabulary_ #Contains word and its frequency
y= pd.get_dummies(msg['Label'])
y= y.iloc[:,1].values

# Creating or TFIDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(max_features=5000)
X = tf.fit_transform(corpus).toarray()
word_freq = tf.vocabulary_ #Contains word and its frequency
idf = tf.idf_ #Contains
y= pd.get_dummies(msg['Label'])
y= y.iloc[:,1].values

#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

#Training the model using Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)

#Prediction
y_pred = spam_detect_model.predict(X_test) # 0 equal ham , 1 equal spam

#Evaluation of our model
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)













