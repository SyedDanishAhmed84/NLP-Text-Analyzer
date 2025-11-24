import nltk
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


st.title("NLP Text Analyzer App")
st.write("Tokenization -  Stemming - Lemmatization - Word Cloud - Sentiment Analysis")

st.sidebar.header('Settings')

token_options=st.sidebar.selectbox(
    'Tokenization',('Word Tokenizer','Sentence Tokenizer'))

stem_option=st.sidebar.checkbox('Stemming')
lemma_option=st.sidebar.checkbox("Lemmatization")
word_option=st.sidebar.checkbox('Word Cloud')

text=st.text_area("Enter your text",height=200)

if st.button("Analyze"):
    if text.strip()=="":
        st.warning("Please enter text to analyze")
    else:
        st.subheader("Cleaned Text")  
        cleaned=text.lower()
        cleaned=re.sub(f"[{re.escape(string.punctuation)}]","",cleaned) 
        st.write(cleaned) 

        # Stop words
        stop_words=set(stopwords.words('english'))
        words=word_tokenize(cleaned)
        filtered_words=[word for word in words if word not in stop_words]

        # Tokenization
        st.subheader("Tokens")
        if token_options == "Word Tokenizer":
            tokens=word_tokenize(cleaned)
        else:
            tokens=sent_tokenize(text)

        st.write(tokens)        

        # Stemming
        if stem_option:
            st.subheader("Stemming")
            ps=PorterStemmer()
            stems=[ps.stem(word) for word in filtered_words]
            st.write(stems)

        # Lemmatization
        if lemma_option:
            st.subheader('Lemmatization')
            lm=WordNetLemmatizer()
            lemmas=[lm.lemmatize(word) for word in filtered_words]
            st.write(lemmas)    

        # Sentiment Analysis
        st.subheader('Sentiment Analysis')
        analyzer=SentimentIntensityAnalyzer()
        scores=analyzer.polarity_scores(text)
        st.write(scores)

        # Word Cloud
        if word_option:
            st.subheader('Word Cloud')
            wc_text = " ".join(filtered_words)
            wc=WordCloud(width=800,height=400).generate(text)
            fig,ax=plt.subplots(figsize=(10,5))
            ax.imshow(wc,interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
       