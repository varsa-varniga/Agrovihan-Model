import joblib
import pandas as pd
# Load the model from the 'model' folder
model = joblib.load('model/agrovihan_model.pkl')

# Example: Make predictions with new data
new_data = pd.DataFrame({
    'crop_type': [0],  # Example: Maize encoded as 0
    'phase': [0],      # Example: Vegetative encoded as 0
    'temp (°C)': [33],
    'humidity (%)': [75],
    'rain (mm)': [28],
    'wind_speed (km/h)': [11]
})

# Predict the suggestion
prediction = model.predict(new_data)

# Example: Map the prediction back to the original suggestion
print(prediction)
