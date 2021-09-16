import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import pprint
import warnings
import tempfile
from io import StringIO
from PIL import  Image
from rake_nltk import Rake
import spacy
import spacy_streamlit
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import vaderSentiment
from collections import Counter
#import en_core_web_sm
from nltk.tokenize import sent_tokenize
from matplotlib import pyplot as plt
from plotly import graph_objs as go

# Warnings ignore 
warnings.filterwarnings(action='ignore')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

#importing the custom module
import sentiment_analyzer as nlp

uploaded_file = st.file_uploader("Upload your sentiment analysis file.(.csv/.txt files only!)", type=["csv",'txt','xlsx'])
#st.cache
df = pd.read_csv(uploaded_file)
df = df.drop(columns =['user_location'], axis = 1)
df.fillna(value = {'text':' '},inplace = True)
df['text'] = df['text'].apply(nlp.clean_text)
df['Subjectivity'] = df['text'].apply(nlp.getSubjectivity)
df['Polarity'] = df['text'].apply(nlp.getPolarity)
df['Analysis'] = df['Polarity'].apply(nlp.getScore)
df['Vader Sentiment'] = df['Analysis'].apply(nlp.vadersentimentanalysis)
df['Vader_Analysis'] = df['Vader Sentiment'].apply(nlp.getScore)


st.title('Elections Sentiment Analyzer')
st.sidebar.markdown('[The Alpha Team]\
                    (https://https://github.com/Joykareko/The-Alpha-Team-DS/)')

option = st.sidebar.selectbox('Navigation', 
["Home",
 "Keyword Sentiment Analysis", 
 "Word Cloud", 
 "Sentiment Prediction"])

st.set_option('deprecation.showfileUploaderEncoding', False)

if option == 'Home':
    st.image('image 2.png',width = 500)
    st.write('When it comes to elections, can we be able to predict outcomes \t'
    'before the elections?\t This is an App that helps you analyze Elections Sentiments.\n' 
    '\t This App helps predict these sentiments using \t Rule Based'
    '\t Natural Language Processing Methods and \t Machine Learning Methods for Text Based Analysis.'
    )
if option == 'Keyword Sentiment Analysis':
    
    st.sidebar.markdown('**How to export a sentiments file?**')
    st.sidebar.text('Follow the steps 👇:')
    st.sidebar.text('1) Collate the sentiments in csv file.')
    st.sidebar.text('2) Tap options > More > Upload file.')
    st.sidebar.text('3) Choose a Rule Based Approach(either Textblob or Vader.')
    st.sidebar.markdown('*You are set to go 😃*.')
    st.sidebar.subheader('**FAQs**')
    st.sidebar.markdown('**Is my uploaded data private?**')
    st.sidebar.markdown('The data you upload is not saved anywhere on this site or any 3rd party site.')


    
    if uploaded_file is not None:
       st.write(type(uploaded_file))
       #st.dataframe(df)
       
    
	# Model Selection 
    model_select = st.selectbox("Choose a Rule Based Approach", ["TextBlob", "Vader"])

    st.button("Analyze")
		
		# Load the model 
    if model_select == "TextBlob":
        st.bar_chart(df['Analysis'].value_counts())
        plt.title('Sentiment Analysis')
        plt.xlabel('Value')
        plt.ylabel('Count')
        #plt.show()
        
         
      
    if model_select == "Vader":
       st.bar_chart(df['Vader_Analysis'].value_counts())
       plt.title('Sentiment Analysis')
       plt.xlabel('Value')
       plt.ylabel('Count')
       
       