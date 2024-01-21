#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from threading import Thread
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load the dataset using a relative path
file_path = os.path.join(os.getcwd(), 'data', 'laptop_price.csv')
laptop_data = pd.read_csv(file_path)

# Load the trained model using pickle
with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the ColumnTransformer
with open('column_transformer.pkl', 'rb') as transformer_file:
    column_transformer = pickle.load(transformer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
            df = pd.DataFrame(data, index=[0])
        else:
            data = request.form.to_dict()
            df = pd.DataFrame([data])

        # Apply the same preprocessing as during training
        df_transformed = column_transformer.transform(df)

        prediction = model.predict(df_transformed)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f"An error occurred during prediction: {str(e)}"}), 500

def run_app():
    app.run(port=6969, debug=True, use_reloader=False, threaded=True)

flask_thread = Thread(target=run_app)
flask_thread.start()


# In[ ]:




