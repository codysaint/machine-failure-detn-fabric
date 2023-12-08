# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:26:39 2020

"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import os
import base64

import pandas as pd
import webbrowser
import pickle
import json

import numpy as np

# Get the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))


app = Flask(__name__)
CORS(app, support_credentials=True)
app.config['SECRET_KEY'] =  os.urandom(24)


output_file = os.path.join(dir_path, 'output', 'predicted_df.csv')

# Construct the full path to the model file

# model1_filename = os.path.join(dir_path, 'models', 'LR_model1.pickle')
# model2_filename = os.path.join(dir_path, 'models', 'LR_model2.pickle')
# model3_filename = os.path.join(dir_path, 'models', 'RF_model3.pkl')

model3_filename = os.path.join(dir_path, 'models', 'RF_model4.pkl')

# model3_filename = os.path.join(dir_path, 'models', 'RF_model5.pkl')

print("model filename: ", model3_filename)

# with open(model1_filename, 'rb') as file:
#     model1 = pickle.load(file) 
    
# with open(model2_filename, 'rb') as file:
#     model2 = pickle.load(file) 
    
with open(model3_filename, 'rb') as file:
    model3 = pickle.load(file)
    
# with open(model4_filename, 'rb') as file:
#     model4 = pickle.load(file)
    
def risk_prob(dataset, model):
    probabilities_output = model3.predict_proba(dataset)[:, 1]
 
    # Create a new DataFrame with the rounded predicted probabilities
    rounded_probabilities = np.round(probabilities_output, 2)
    
    return rounded_probabilities

def predict_results(dataset, model):
    pred_df = dataset.copy()
    result = 'Shape of dataset : {}'.format(dataset.shape)
    print(result)
    
    ##### predict the test data
    if(model == 'model3'):
        y_pred= model3.predict(dataset)
        risk_probability = risk_prob(dataset, model)
    
    # elif(model == 'model2'):
    #     y_pred= model2.predict(dataset)
    # elif(model == 'model3'):
    #     y_pred= model3.predict(dataset)
    else:
        response = {'status' : False, 'error': 'Please provide input for engine!!'}
        return response
    
    print(y_pred.shape)
    
    pred_df['Y_Failure_Prediction'] = y_pred
    pred_df['risk_prob'] = risk_probability
    pred_df['Prod_Quality'] = pred_df['Type'].map({0: 'Low', 1: 'Med', 2: 'High'})
    
    pred_df.drop(columns=['Type'], inplace=True)
    
    # cols = list(pred_df.columns)
    # cols = [col for col in cols if col != 'Y_Failure_Prediction']
    # pred_df = pred_df[cols + ['Y_Failure_Prediction']]
      
    pred_df = pred_df.sample(frac=0.01).reset_index(drop=True)  
    
    # print(pred_df.head())
    pred_df.to_csv(output_file, index= False)
    response = {'status': True, 'pred_df' : pred_df.to_dict('records')}
    
    # print(response)
    
    return response



############################################################################################
########### API for each process
############################################################################################


##########################################################################################
##########################################################################################
#GET API
@app.route('/', methods = ['GET'])
@cross_origin(supports_credentials=True)
def get_index_page():
	return render_template('index.html')


############################################################################################
########### POST API for predicting
############################################################################################
#POST API with body parameters as file
@app.route("/predict", methods=['POST'])
@cross_origin(supports_credentials=True)
def explore_dataset_api():
    request_dict = request.json
    #filename = request_dict['filename']
    model = request_dict['model']
    print ("model : ",model)
    
    filedata = request_dict['filedata']
    filename = os.path.join(dir_path, 'output', 'test.csv')
    data = base64.b64decode(filedata.split(",")[1])

    with open(filename, "wb") as f:
        f.write(data)
        f.close()
    print ("Filename : ",filename)
    
    dataset = pd.read_csv(filename)
    response = predict_results(dataset, model)
    return jsonify(response)
    
############################################################################################
########### Main Function
############################################################################################
if __name__ == '__main__':
    # chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    # webbrowser.get(chrome_path).open('file://' + os.path.realpath('view/index.html'))
    # app.run(port=5000)
    app.run(host='0.0.0.0')
   
