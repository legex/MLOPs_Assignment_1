import joblib
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('xgboost_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = data['features']

    # Predict using the loaded model
    prediction = model.predict([features])

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
