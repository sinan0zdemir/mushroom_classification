from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for web app

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../../models/mushroom_mobilenet_finetuned.h5')
model = None

# Mushroom class names (adjust these to match your actual classes)
CLASS_NAMES = [
    'Amanita Citrina', 'Amanita Muscaria', 'Amanita Pantherina', 'Amanita Phalloides', 'Amanita Rubescens',
    'Armillaria Mellea', 'Auricularia Auricula Judae', 'Auricularia Mesenterica', 'Boletus Edulis', 'Clathus Ruber',
    'Coprinus Comatus', 'Cyclocybe Cylindracea', 'Fomitopsis Pinicola', 'Laetiporus Sulphureus', 'Lycoperdon Perlatum',
    'Macrolepiota Procera', 'Phallus Impudicus', 'Schizophyllyum Commune', 'Trametes Versicolor', 'Volvopluteus Gloiocephalus'
]

def load_model():
    """Load the trained model"""
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at: {MODEL_PATH}")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype('float32') / 255.0
    
    return image_array

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'API is running',
        'model_loaded': model is not None,
        'classes': len(CLASS_NAMES)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict mushroom class from uploaded image"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'class': CLASS_NAMES[idx],
                'confidence': float(predictions[0][idx]),
                'percentage': f"{float(predictions[0][idx]) * 100:.2f}%"
            }
            for idx in top_3_idx
        ]
        
        return jsonify({
            'success': True,
            'predicted_class': CLASS_NAMES[predicted_class_idx],
            'confidence': confidence,
            'confidence_percentage': f"{confidence * 100:.2f}%",
            'top_predictions': top_3_predictions
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of all mushroom classes"""
    return jsonify({
        'classes': CLASS_NAMES,
        'total_classes': len(CLASS_NAMES)
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)