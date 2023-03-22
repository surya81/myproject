import numpy as np
import pandas as pd 
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
        print(request.form)
        quantity_tons = float(request.form["quantity_tons"])
        customer = float(request.form["customer"])
        country = float(request.form["country"])
        status = float(request.form["status"])
        product_ref = float(request.form["product_ref"])
        Area= float(request.form["Area"])
        Date_difference = float(request.form["Date difference"])
        values = np.array([[ quantity_tons,customer, country,status,product_ref,Area,Date_difference ]]).reshape(1,-1)
        print(values)
        prediction =model.predict(values)
        return render_template('index.html', prediction_text='Selling price cost would be  {:.2f}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
