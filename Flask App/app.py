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
from process_input import main

app = Flask(__name__)
app.secret_key = "secret keey"

@app.route('/', methods=['GET', 'POST'])
def index():    
    if request.method == 'POST':

        news = request.form["news"]
        value_sequence = main(news)

        with open('fakeNews.pkl' , 'rb') as file2:
            classifier = pickle.load(file2)
        
        result = classifier.predict(value_sequence)

        if result < 0.5 :
            print("This newsline seems genuine :)")
            flash("This newsline seems genuine :)")
        elif result > 0.5 and result < 1.5:
            print("This newsline seems suspicious, tread carefully")
            flash("This newsline seems suspicious, tread carefully")
        elif result > 1.5:
            print("This newsline seems fake. Take it with a pinch of salt!")
            flash("This newsline seems fake. Take it with a pinch of salt!")

        return render_template('index.html', newsline=news)

    else:
        return render_template('index.html')