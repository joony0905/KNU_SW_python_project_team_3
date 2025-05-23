# app/main.py
from flask import Flask, render_template, request, jsonify
from app.utils.file_io import *
from app.utils.preprocessing import *
from app.model.predict import *
from app.model.data_loader import *
from app.utils.feedback import *


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
        prediction, label = predict_with_selected_model(processed_input, model_choice, return_label=True)  # <- label도 리턴하도록
        return render_template('index.html',
                               input_text=user_input,
                               prediction_text=prediction.replace('\n', '<br>'),
                               prediction_label=label,
                               model_choice=model_choice)

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    result = process_feedback(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5050)
