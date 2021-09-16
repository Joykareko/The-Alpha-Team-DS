# Standard Libraries
import os 
import re 
import string 
import numpy as np
import pandas as pd
from collections import Counter

# Text Processing Library 
import nltk 
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from textblob import TextBlob
#from wordcloud import WordCloud
from gensim import utils
import streamlit as st
import pprint
import gensim
import gensim.downloader as api
import warnings
import spacy
from spacy import displacy
from pathlib import Path
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span
import tempfile
warnings.filterwarnings(action='ignore')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Data Visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns
import spacy_streamlit
from PIL import Image

# Constants 
STOPWORDS = stopwords.words('english')
STOPWORDS + ['said']

#cleaning the data
#Text cleaning function 
def clean_text(text):
    '''
        Function which returns a clean text 
    '''  
    # Lower case 
    text = text.lower()  
    # Remove numbers
    text = re.sub(r'\d','', text)
    text = re.sub(r'@[A-Za-z0-9]+','',text) #removes @mentions and substitutes with an empty string
    text = re.sub(r'@_','',text)
    text = re.sub(r'#','', text) #removes #tags
    text = re.sub(r'RT[\s]+','',text) #removes the RT
    text = re.sub(r'https?:\/\/\S+','',text)

  
    # Replace \n and \t functions 
    text = re.sub(r'\n','', text)
    text = text.strip()
    
    # Remove punctuations
    text = text.translate(str.maketrans('','', string.punctuation))
    
    # Remove Stopwords and Lemmatise the data
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in STOPWORDS]
    text = ' '.join(text)
    return text
#subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#creating a function that shows the polarity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

#function that computes the negative/positive/neutral sentiments
def getScore(score):
    if score <0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else: 
        return 'Positive'
#summary on vader
analyzer = SentimentIntensityAnalyzer()
def vadersentimentanalysis(text):
    vs = analyzer.polarity_scores(text)
    return vs['compound']
  

