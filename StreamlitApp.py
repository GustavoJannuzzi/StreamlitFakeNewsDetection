#  STREAMLIT DEPLOY
#  --- --- --- ---
#  Machine Learning Microbe Prediction:

#import libs
import nltk
from string import punctuation
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from sklearn.svm import LinearSVC

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

# ______________________
#   PREDICTIVE MODEL 
#------------------------

#Read data
fake_news = pd.read_csv('https://raw.githubusercontent.com/GustavoJannuzzi/StreamlitFakeNewsDetection/master/Fake.csv')
true_news = pd.read_csv('https://raw.githubusercontent.com/GustavoJannuzzi/StreamlitFakeNewsDetection/master/True.csv')
fake_news['target'] = 0
true_news['target'] = 1

data0 = pd.concat([fake_news, true_news])
data0 = data0.reset_index(drop=True)

#data clean
data1 = data0.copy()
data1 = data1.drop(['title', 'subject', 'date'], axis=1)

data2 = data1.copy()
data2 = shuffle(data2, random_state=0)
data2 = data2.reset_index(drop=True)

def cleaner(text):
    
    text = text.lower()
    
    text = ''.join(c for c in text if not c.isdigit()) #remove digits
    text = ''.join(c for c in text if c not in punctuation) #remove all punctuation
    
    stop_words = stopwords.words('english') # removes words which has less meaning 
    text = ' '.join([w for w in nltk.word_tokenize(text) if not w in stop_words])
    
    wordnet_lemmatizer = WordNetLemmatizer() # with use of morphological analysis of words
    text = [wordnet_lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)]
    
    text = " ".join(w for w in text)
    return text

data3 = data2.copy()
data3['text'] = data3['text'].apply(cleaner)

#Create the model
X = data3['text']
y = data3['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer()
clf = LinearSVC(C=10)

pipeline = Pipeline([
('tfidf', tfidf),
('classifier', clf),  
    ])
pipeline.fit(X_train, y_train) 

### Create a Pickle file 
import pickle
pickle_out = open("fake_news_detect.pkl","wb")
pickle.dump(pipeline, pickle_out)
pickle_out.close()


# _______________________
#       STREAMLIT 
# ------------------------
# Input bar 
news = st.text_input("Paste the News")

# If button is pressed
if st.button("DETECT"):
    
    # Unpickle classifier
    clf = joblib.load("fake_news_detect.pkl")
    
    # Store inputs into dataframe
    X = [news]
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    if prediction == 1:
        st.subheader('fake news')
    else:
        st.subheader('real news')
