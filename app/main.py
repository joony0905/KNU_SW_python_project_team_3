# app/main.py
from flask import Flask, render_template, request, jsonify
from utils.file_io import *
from utils.preprocessing import *
from model.predict import *
from model.data_loader import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['message']
        processed_input = preprocess_user_input(user_input)
        model_choice = request.form['model_choice']
        prediction = predict_with_selected_model(processed_input, model_choice)
        return render_template('index.html', input_text=user_input, prediction_text=prediction.replace('\n', '<br>'))

if __name__ == '__main__':
    app.run(debug=True)