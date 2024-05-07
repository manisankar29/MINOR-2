# xray_prediction.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained X-ray model
xray_model = load_model('xray_model.h5')

def predict_xray(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values

    # Make prediction
    prediction = xray_model.predict(img_array)
    
    # Placeholder logic for result extraction
    # Replace this with your actual result extraction logic based on your model's output
    if prediction[0][0] > 0.5:
        result = "Positive for pneumonia"
    else:
        result = "Negative for pneumonia"

    return result
