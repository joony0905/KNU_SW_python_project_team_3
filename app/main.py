# app/main.py
from flask import Flask, render_template, request, jsonify
from utils.file_io import *
from utils.preprocessing import *
from model.predict import *
from model.data_loader import *
from utils.feedback import *


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['message']
        model_choice = request.form['model_choice']
        prediction, label = predict_with_selected_model(user_input, model_choice, return_label=True)  # 함수 내에서 전처리하게 변경
        return render_template('index.html',
                               input_text=user_input,
                               prediction_text=prediction.replace('\n', '<br>'),
                               prediction_label=label,
                               model_choice=model_choice
                               )

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    result = process_feedback(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5050)
