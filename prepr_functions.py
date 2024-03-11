import pandas as pd
import numpy as np
# We'll be working with df_month
#Directory
import os
# Storing Data
import csv
# Tracking loading progress
from tqdm import tqdm
tqdm.pandas()
# Preprocessing
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer 
from nltk.stem import PorterStemmer
#nltk.download('stopwords')
from nltk.corpus import stopwords # Add MODERATOR TO THE STOPWORDS LIST
#!python -m spacy download en_core_web_sm
import spacy
import pickle
sp = spacy.load('en_core_web_sm')
#Regex
import re

#Plotting
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


#getting a library of stopwords and defining a lemmatizer
porter=SnowballStemmer("english")
lmtzr = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))


################## Functions - Cleaning


def correct_quoted_contractions(text):
    # Define a regular expression pattern to capture quoted expressions
    pattern = r'"([^"]*)"'
    
    # Define a replacement function to correct contractions
    def replace(match):
        return match.group(1).replace(' ', "'")
    
    # Apply the replacement function to the text using re.sub
    corrected_text = re.sub(pattern, replace, text)
    
    return corrected_text


def list_to_sentence(term_list):
    return ' '.join(term_list)



################## Functions - Preprocessing


#Returns words without any special 
def strip(word):
    mod_string = re.sub(r'\W+', '', word)
    return mod_string

#the following leaves in place two or more capital letters in a row
#will be ignored when using standard stemming
def abbr_or_lower(word):
    if re.match('([A-Z]+[a-z]*){2,}', word):
        return word
    else:
        return word.lower()

#modular pipeline for stemming, lemmatizing and lowercasing
#note this is NOT lemmatizing using grammar pos
    
def tokenize(text, modulation):
    stop_words.add('MODERATOR')
    if modulation<2:
        tokens = re.split(r'\W+', text)
        stems = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            lowers=abbr_or_lower(token)
            if lowers not in stop_words:
                if re.search('[a-zA-Z]', lowers):
                    if modulation==0:
                        stems.append(lowers)
                    if modulation==1:
                        stems.append(porter.stem(lowers))
    else:
        sp_text=sp(text)
        stems = []
        lemmatized_text=[]
        for word in sp_text:
            lemmatized_text.append(word.lemma_)
        stems = [abbr_or_lower(strip(w)) for w in lemmatized_text if (abbr_or_lower(strip(w))) and (abbr_or_lower(strip(w)) not in stop_words)]
    return " ".join(stems)


def txtprocess_tok(corpus,col,mod):
    text_preproc = (
    corpus[col]
    .astype(str)
    .progress_apply(lambda row: tokenize(row, mod))
    )
    return text_preproc


def vectorize(tokens, vocab):
    vector=[]
    for w in vocab:
        vector.append(tokens.count(w))
    return vector


################## Functions - Data Exploration

def freq_table(dataframe,group,variable,ascend):
# Creating a Frequency table
    var = dataframe.groupby(group)[variable].count().reset_index()
    var = var.sort_values(by=variable,ascending=ascend)
    var['total'] = var[variable].sum()
    var['perc'] = (var['reply_sentence'] / var['total']).round(2)
    var['accum'] = var['perc'].cumsum()
    return var




################################ Other Functions
# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # Set norms to 1 if they are 0 to avoid division by zero
    if norm_vec1 == 0:
        norm_vec1 = 1
    if norm_vec2 == 0:
        norm_vec2 = 1
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def normalize_vectors(dtm):
    # Calculate the norm of each article vector
    norms = np.linalg.norm(dtm, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1
    # Normalize each article vector to length 1
    normalized_dtm = dtm / norms
    return normalized_dtm


def calculate_avg_and_avg_cosine_similarity(normalized_dtm, labels):
    avg_vectors = {}
    avg_cosine_similarities = {}
    for label in np.unique(labels):
        print("Doing label:", label)
        # Filter the articles by label
        articles_with_label = normalized_dtm[labels == label]
        # Calculate the average vector for the label
        avg_vector = np.mean(articles_with_label, axis=0)
        avg_vectors[label] = avg_vector
        # Calculate the cosine similarity of each article vector to the average vector
        cosine_similarities = [cosine_similarity(article_vector, avg_vector) for article_vector in articles_with_label]
        # Calculate the average cosine similarity for the label
        avg_cosine_similarities[label] = np.mean(cosine_similarities)
        print("Average cosine similarity to average vector:", avg_cosine_similarities[label])
        print("     ")
    return avg_vectors, avg_cosine_similarities