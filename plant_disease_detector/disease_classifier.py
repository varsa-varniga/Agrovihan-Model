import numpy as np
from PIL import Image
import tensorflow as tf

# Load pre-trained model
model = tf.keras.models.load_model("models/mobilenetv2_model.h5")

# Define the class names
class_names = [
    'Paddy_Rice_Blast', 'Paddy_Bacterial_Leaf_Blight', 'Blackgram_Yellow_Mosaic_Virus',
    'Blackgram_Powdery_Mildew', 'Groundnut_Tikka_Leaf_Spot', 'Groundnut_Rust',
    'Banana_Panama_Wilt', 'Banana_Sigatoka_Leaf_Spot', 'Tomato_Early_Blight',
    'Tomato_Leaf_Curl_Virus', 'Sugarcane_Red_Rot', 'Sugarcane_Smut'
]

# Load and preprocess the image
img_path = 'tomato leafcurl2.png'
img = Image.open(img_path).convert("RGB")
img = img.resize((224, 224))

# Preprocess image for model input
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Get model prediction
prediction = model.predict(img_array)

# Find the predicted class
predicted_class_index = np.argmax(prediction)  # Get the index of the highest probability
predicted_class = class_names[predicted_class_index]  # Map index to disease name

# Print the predicted class index and name for debugging
print(f"Predicted Class Index: {predicted_class_index}")
print(f"Predicted Disease Name: {predicted_class}")
