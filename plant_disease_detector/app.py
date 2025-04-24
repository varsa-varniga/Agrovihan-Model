from flask import Flask, request, jsonify 
import tensorflow as tf 
import numpy as np
from PIL import Image
from flask_cors import CORS

# Import organic_solutions dictionary directly
organic_solutions = {
    "01_healthy_paddy": {
        "solution": """
        Apply **Trichoderma** to enhance soil health and prevent fungal infections.
        Use **neem oil** and **garlic extract** to control pests naturally.
        Practice crop rotation and use **vermicompost** to improve soil fertility.
        """
    },
    "02_leaf_blast_paddy": {
        "solution": """
        Apply **cow urine** mixed with **neem oil** to prevent bacterial infections.
        Incorporate proper drainage and avoid waterlogging.
        Use **Trichoderma** as a preventive measure for fungal growth.
        """
    },
    "03_bacterial_leaf_blight_paddy": {
        "solution": """
        Use **neem oil** or **garlic extract** to treat and prevent bacterial infections.
        Apply **cow urine** for additional protection against pathogens.
        Improve drainage to reduce the spread of bacteria and maintain plant health.
        """
    },
    "04_healthy_banana": {
        "solution": """
        Regularly apply **neem oil** for pest control.
        Use **vermicompost** and **compost** to enhance soil nutrients.
        Mulch around the banana plant base to retain moisture and reduce weed growth.
        """
    },
    "05_cordana_banana": {
        "solution": """
        Apply **Trichoderma** to boost soil health and fight fungal infections.
        Use **neem oil** and **garlic extract** for pest control.
        Maintain proper spacing for healthy growth and airflow between plants.
        """
    },
    "06_sigatoka_banana": {
        "solution": """
        Use a **neem oil** and **baking soda** mixture to treat fungal diseases.
        Remove and destroy any infected leaves immediately to prevent the spread.
        Apply **Trichoderma** to boost overall soil health and reduce fungal infections.
        """
    },
    "07_Healthy_sugarcane": {
        "solution": """
        Apply **organic compost** and **cow dung** around the base of sugarcane plants.
        Use **Trichoderma** to improve the microbial health of the soil.
        Ensure proper irrigation, avoiding waterlogging to keep the roots healthy.
        """
    },
    "08_Mosaic_sugarcane": {
        "solution": """
        Use **neem oil** or **garlic extract** to control aphids and other insect vectors.
        Regularly remove and destroy infected leaves to prevent the spread of the virus.
        Avoid planting sugarcane in areas with a history of the disease.
        """
    },
    "09_RedRot_sugarcane": {
        "solution": """
        Apply **Trichoderma** to combat fungal infections in the soil.
        Use **neem oil** or **cow urine** around the base of sugarcane plants to protect against fungal infections.
        Ensure proper drainage and avoid waterlogging to prevent fungal growth.
        """
    },
    "10_healthy_leaf_groundnut": {
        "solution": """
        Use **cow dung slurry** or **garlic extract** to prevent common pests.
        Apply **Trichoderma** to improve soil health and prevent fungal infections.
        Ensure good plant spacing and proper irrigation to avoid stress.
        """
    },
    "11_early_leaf_spot_groundnut": {
        "solution": """
        Spray **neem oil** or **baking soda** on the affected areas to control leaf spot.
        Apply **Trichoderma** to improve soil health and boost resistance against fungal infections.
        Remove infected plant parts immediately to reduce the spread of disease.
        """
    },
    "12_Rust_groundnut": {
        "solution": """
        Apply **neem oil** for controlling rust and other fungal diseases.
        Use **Trichoderma** to improve soil health and boost resistance.
        Practice crop rotation to reduce the chances of disease recurrence.
        """
    },
    "13_Healthy_blackgram": {
        "solution": """
        Use **neem oil** and **garlic extract** for pest control.
        Apply **vermicompost** and **compost** to nourish the soil.
        Use **Trichoderma** for improving soil health and preventing fungal infections.
        """
    },
    "14_Yellow_Mosaic_blackgram": {
        "solution": """
        Use **neem oil** to control aphids and other insect vectors of the disease.
        Practice crop rotation and use resistant varieties of blackgram.
        Remove infected plants immediately to prevent the virus spread.
        """
    },
    "15_Powdery_Mildew_blackgram": {
        "solution": """
        Apply a **neem oil** or **baking soda** mixture to treat powdery mildew.
        Ensure proper air circulation around plants to reduce humidity and the spread of mildew.
        Use **Trichoderma** to help improve soil health and fight fungal infections.
        """
    },
    "16_tomato_healthy": {
        "solution": """
        Use **neem oil** for natural pest control.
        Apply **vermicompost** and **compost** to provide essential nutrients.
        Mulch around the plant base to retain moisture and reduce weed growth.
        """
    },
    "17_Tomato_Yellow_Leaf_Curl_Virus": {
        "solution": """
        Use **neem oil** to control aphids and other insect vectors.
        Remove and destroy infected plants immediately to prevent the spread of the virus.
        Practice proper spacing and crop rotation to minimize disease spread.
        """
    },
    "18_Early_blight_tomato": {
        "solution": """
        Spray a **neem oil** solution to treat fungal infections.
        Remove infected leaves and dispose of them away from the garden.
        Apply **Trichoderma** to promote beneficial soil microorganisms.
        """
    }
}

# Initialize Flask app
app = Flask(__name__) 
CORS(app)   # Enable CORS

# Load pre-trained model
model = None  # Will be loaded before first request

# Class names - EXACTLY match your folder names used in training (in correct order)
class_names = [
    '01_healthy_paddy',
    '02_leaf_blast_paddy',
    '03_bacterial_leaf_blight_paddy',
    '04_healthy_banana',
    '05_cordana_banana',
    '06_sigatoka_banana',
    '07_Healthy_sugarcane',
    '08_Mosaic_sugarcane',
    '09_RedRot_sugarcane',
    '10_healthy_leaf_groundnut',
    '11_early_leaf_spot_groundnut',
    '12_Rust_groundnut',
    '13_Healthy_blackgram',
    '14_Yellow_Mosaic_blackgram',
    '15_Powdery_Mildew_blackgram',
    '16_tomato_healthy',
    '17_Tomato_Yellow_Leaf_Curl_Virus',
    '18_Early_blight_tomato'
]

# Plant information dictionary
plant_info = {
    "paddy": "Rice (Paddy) is a staple food crop cultivated in flooded fields. It requires proper water management and is susceptible to various fungal and bacterial diseases in humid conditions.",
    "banana": "Banana plants are large herbaceous plants that produce nutritious fruits. They thrive in tropical climates and require regular watering and nutrient-rich soil.",
    "sugarcane": "Sugarcane is a tall perennial grass used for sugar production. It requires plenty of sunlight, regular irrigation, and well-drained, nutrient-rich soil.",
    "groundnut": "Groundnut (Peanut) is a legume crop grown for its edible seeds. It improves soil fertility through nitrogen fixation and thrives in well-drained sandy loam soils.",
    "blackgram": "Blackgram is a pulse crop rich in protein. It's drought-tolerant and helps in soil nitrogen fixation, making it valuable in crop rotation practices.",
    "tomato": "Tomato plants are popular vegetables that need full sunlight and moderate watering. They're susceptible to various diseases, especially in humid conditions."
}

# Helper function to extract plant type from class name
def extract_plant_type(class_name):
    for plant in plant_info.keys():
        if plant in class_name.lower():
            return plant
    return "unknown"

# Get solution from organic_solutions dictionary
def get_disease_solution(prediction):
    # Get the predicted class name
    class_index = np.argmax(prediction)
    raw_class_name = class_names[class_index]
    
    # Extract plant type
    plant_type = extract_plant_type(raw_class_name)
    plant_description = plant_info.get(plant_type, "No additional information available for this plant.")
    
    # Format disease name for display
    if "_" in raw_class_name and raw_class_name[0].isdigit():
        parts = raw_class_name.split("_")
        if len(parts) > 1:
            formatted_disease = " ".join(word.capitalize() for word in "_".join(parts[1:]).replace("_", " ").split())
    else:
        formatted_disease = " ".join(word.capitalize() for word in raw_class_name.replace("_", " ").split())
    
    # Get solution directly from organic_solutions using the raw class name
    solution = organic_solutions.get(raw_class_name, {}).get("solution", "")
    if not solution:
        solution = "No specific solution available for this disease."
    else:
        # Clean up the solution text (remove extra whitespace)
        solution = "\n".join(line.strip() for line in solution.split("\n") if line.strip())
    
    return formatted_disease, solution, plant_description

# Preprocess uploaded image
def preprocess_image(file):
    try:
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, None
    except Exception as e:
        return None, str(e)

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.content_type.startswith('image/'):
        return jsonify({"error": "Uploaded file is not an image"}), 400
    
    img_array, error_message = preprocess_image(file)
    if img_array is None:
        return jsonify({"error": f"Invalid image file: {error_message}"}), 400
    
    prediction = model.predict(img_array)
    disease_name, solution, additional_info = get_disease_solution(prediction)
    
    # Print debug info
    class_index = np.argmax(prediction)
    raw_class_name = class_names[class_index]
    print(f"Raw class name: {raw_class_name}")
    print(f"Disease name: {disease_name}")
    print(f"Solution found: {'Yes' if solution != 'No specific solution available for this disease.' else 'No'}")
    
    return jsonify({
        "disease": disease_name,
        "solution": solution,
        "additional_tips": additional_info
    })

# Run the app
if __name__ == '__main__':
    @app.before_first_request
    def load_model():
        global model
        model = tf.keras.models.load_model("models/mobilenetv2_model.h5")
    
    app.run(debug=True, port=5002)
