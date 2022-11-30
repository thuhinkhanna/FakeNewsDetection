import re
import pickle
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score
from flask import Flask, request, redirect, flash, url_for, session, jsonify
from flask import render_template

def normalize(data):
    normalized = []
    for i in data:
        i = i.lower()
        # remove urls
        i = re.sub('https?://\S+|www\.\S+', '', i)
        # remove non words and extra spaces
        i = re.sub('\\W', ' ', i)
        i = re.sub('\n', '', i)
        i = re.sub(' +', ' ', i)
        i = re.sub('^ ', '', i)
        i = re.sub(' $', '', i)
        normalized.append(i)
    return normalized

def main(news):

    l = []
    l.append(news)
    normalized = normalize(l)

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    tokenized = tokenizer.texts_to_sequences(normalized)
    preprocessed = tf.keras.preprocessing.sequence.pad_sequences(tokenized)

    return preprocessed