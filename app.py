#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request


# In[2]:


app = Flask(__name__, template_folder='templates')
@app.route('/')
def student():
    return render_template("home.html")
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,4)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return round(result[0],2)
@app.route('/',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = float(ValuePredictor(to_predict_list))
        return render_template("home.html",result_text = 'Prediksi Gaji Sebesar Rp. {}'.format(result))
if __name__ == '__main__':
    app.run(debug = True)


# In[ ]:




