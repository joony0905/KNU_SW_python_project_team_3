# app/utils/feedback.py
import json
import os

def save_feedback(message, label):
    with open("feedback_data.json", "a", encoding="utf-8") as f:
        json.dump({"message": message, "label": label}, f, ensure_ascii=False)
        f.write("\n")


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data","raw")
os.makedirs(DATA_DIR, exist_ok=True)
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_data.json")

def process_feedback(data):
    message = data['message']
    feedback = data['feedback']
    prediction = data.get('prediction', 'unknown')
    model = data.get('model', 'unknown')

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        json.dump({
            "message": message,
            "feedback": feedback,
            "prediction": prediction,
            "model": model
        }, f, ensure_ascii=False)
        f.write("\n")

    return {"status": "success"}
