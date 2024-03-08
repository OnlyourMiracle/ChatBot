# -*- coding: utf-8 -*- 
import os
from datapreprocess import preprocess
import train_eval
import fire
from QA_data import QA_test
from config import Config
from flask import Flask, request, render_template


# Define a function to respond based on keywords
def respond_to_user(input_text):
    opt = Config()
    '''
    for k, v in kwargs.items(): #设置参数
        setattr(opt, k, v)   
        '''

    searcher, sos, eos, unknown, word2ix, ix2word = train_eval.test(opt)

    if os.path.isfile(opt.corpus_data_path) == False:
        preprocess()

    output_words = train_eval.output_answer(input_text, searcher, sos, eos, unknown, opt, word2ix, ix2word)
    
    return output_words


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("indexV4.html")  # You'll create this HTML file for the interface

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    return respond_to_user(user_text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
