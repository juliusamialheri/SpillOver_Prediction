from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "students.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Support both JSON and Form data
        if request.is_json:
            data = request.get_json()
            features = [
                float(data.get('1', 0)),
                float(data.get('2', 0)),
                float(data.get('3', 0)),
                float(data.get('4', 0)),
                float(data.get('5', 0))
            ]
        else:
            features = [
                float(request.form.get('1', 0)),
                float(request.form.get('2', 0)),
                float(request.form.get('3', 0)),
                float(request.form.get('4', 0)),
                float(request.form.get('5', 0))
            ]

        row_df = pd.DataFrame([features], columns=['FirstGPA', 'SecGPA', 'CGPA', 'NOC', 'TOCU'])
        prediction = model.predict_proba(row_df)
        prob = float(prediction[0][1])
        percent = round(prob * 100, 2)

        result_text = "You have already Spilled Over." if prob >= 0.4 else "You are not in danger of Spill Over."

        if request.is_json:
            return jsonify({
                'probability': prob,
                'percent': percent,
                'result': result_text,
                'status': 'success'
            })

        return render_template('result.html', pred=f'{result_text} ({percent}%)')

    except Exception as e:
        if request.is_json:
            return jsonify({'error': str(e), 'status': 'error'}), 400
        return f"Error: {str(e)}", 400



if __name__ == '__main__':
    app.run(debug=True)
