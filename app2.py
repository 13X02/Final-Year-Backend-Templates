from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load the trained model
model = load_model('model.h5')

# Define class labels
class_labels = ['Basal cell carcinoma (bcc)','Melanocytic nevi (nv)', 'Melanoma (mel)', 'Benign keratosis-like lesions (bkl)',
                 'Actinic keratoses (akiec)', 'Vascular lesions (vas)', 
                'Dermatofibroma (df)']

# Function to preprocess the image
# Function to preprocess the image
def preprocess_image(image):
    # Resize image to match model input size (32x32)
    image = image.resize((32, 32))
    # Convert image to numpy array
    image_array = np.asarray(image)
    # Check if image has 4 channels
    if image_array.shape[-1] == 4:
        # Remove the alpha channel (transparency)
        image_array = image_array[..., :3]
    # Normalize pixel values to range [0, 1]
    image_array = image_array / 255.0
    # Add batch dimension and return
    return np.expand_dims(image_array, axis=0)


# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if request contains file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Read image file
        img = Image.open(file)
        # Preprocess image
        img = preprocess_image(img)
        # Make prediction
        prediction = model.predict(img)
        # Get predicted class label and probability
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)
        return jsonify({'class': predicted_class, 'confidence': str(confidence)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
