from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # Explicitly allow the frontend origin

# Load models and encoders
with open('model/agrovihan_model_rf.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('model/agrovihan_model_xgb.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('model/suggestion_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('model/columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Load pre-trained scaler
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)  # Load the scaler that was fitted on training data

@app.route('/suggest', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        
        # Extract features from the request - match the keys used in frontend
        crop_type = data['crop_type']
        phase = data['phase']
        temperature = data['temp']  # Changed from 'temperature' to match frontend
        humidity = data['humidity']
        rain = data.get('rain', 0)  # Use get() with default in case it's missing
        wind_speed = data['wind_speed']
        
        # Create DataFrame with the same columns as expected by the model
        input_data = pd.DataFrame([{
            'crop_type': crop_type,
            'phase': phase,
            'temperature': temperature,
            'humidity': humidity,
            'rain': rain,
            'wind_speed': wind_speed
        }])
        
        # One-hot encode the features
        input_data_encoded = pd.get_dummies(input_data)
        
        # Ensure all the required columns are in the input data
        missing_cols = set(feature_columns) - set(input_data_encoded.columns)
        for col in missing_cols:
            input_data_encoded[col] = 0
        
        # Reorder columns to match the model's training data
        input_data_encoded = input_data_encoded[feature_columns]
        
        # Normalize the input data using the pre-trained scaler
        input_data_scaled = scaler.transform(input_data_encoded)
        
        # Make prediction using the Random Forest model (or switch to XGBoost)
        prediction = rf_model.predict(input_data_scaled)[0]  # Change to xgb_model if needed
        
        # Decode the prediction back to the original class label
        suggestion = label_encoder.inverse_transform([prediction])[0]
        
        # Return the suggestion as a JSON response
        return jsonify({'suggestion': suggestion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
