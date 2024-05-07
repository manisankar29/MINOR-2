from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained X-ray model
xray_model = tf.keras.models.load_model('xray_model.h5')
# Load the pre-trained MRI model
mri_model = tf.keras.models.load_model('mri_model.h5')
ct_model= tf.keras.models.load_model('ct_model.h5')


# Define class labels for MRI model
#class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Function to preprocess X-ray image
def preprocess_xray_image(image):
    # Resize the image to the required input shape of the model
    resized_image = image.resize((220, 220))
    # Convert the image to RGB mode (in case it's grayscale)
    resized_image = resized_image.convert('RGB')
    # Convert the image to a numpy array
    img_array = np.array(resized_image)
    # Normalize the image
    normalized_image = img_array / 255.0
    # Add batch dimension
    processed_image = np.expand_dims(normalized_image, axis=0)
    return processed_image

# Function to preprocess MRI image
def preprocess_mri_image(image):
    # Resize the image to match the input shape of the model
    resized_image = image.resize((150, 150))
    # Convert the image to RGB mode (in case it's grayscale)
    resized_image = resized_image.convert('RGB')
    # Convert the image to a numpy array
    img_array = np.array(resized_image)
    # Normalize the image
    normalized_image = img_array / 255.0
    # Add batch dimension
    processed_image = np.expand_dims(normalized_image, axis=0)
    return processed_image


# Function to make X-ray prediction
def predict_xray(image):
    # Preprocess the image
    processed_image = preprocess_xray_image(image)
    # Perform prediction
    prediction = xray_model.predict(processed_image)
    # Return prediction
    return prediction
def predict_mri(image):
    # Preprocess the image
    processed_image = preprocess_mri_image(image)
    # Perform prediction
    prediction = mri_model.predict(processed_image)
    # Process prediction result
    binary_prediction = "With Brain Tumor" if prediction[0][0] > 0.5 else "Without Brain Tumor"
    # Return binary prediction
    return binary_prediction


# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for rendering the X-ray page
@app.route('/xray')
def xray():
    return render_template('xray.html')

# Route for handling X-ray predictions
@app.route('/predict_xray', methods=['POST'])
def predict_xray_route():
    # Get the uploaded image
    uploaded_file = request.files['xray_image']
    if uploaded_file.filename != '':
        # Read the image
        img = Image.open(uploaded_file)
        # Perform X-ray prediction
        prediction = predict_xray(img)
        # Process prediction result
        result = "Pneumonia Detected" if prediction[0][0] > 0.5 else "No Pneumonia Detected"
        # Return prediction result as JSON
        return jsonify({'result': result})
    else:
        return jsonify({'error': 'No file uploaded'})

# Route for rendering the MRI page
@app.route('/mri')
def mri():
    return render_template('mri.html')

# Route for handling MRI predictions
@app.route('/predict_mri', methods=['POST'])
def predict_mri_route():
    # Get the uploaded image
    uploaded_file = request.files['mri_image']
    if uploaded_file.filename != '':
        # Read the image
        img = Image.open(uploaded_file)
        # Perform MRI prediction
        prediction = predict_mri(img)
        # Return prediction result as JSON
        return jsonify({'result': prediction})
    else:
        return jsonify({'error': 'No file uploaded'})

# Function to preprocess CT scan image
def preprocess_ct_image(image):
    # Resize the image to match the input shape of the model
    resized_image = image.resize((224, 224))
    # Convert the image to RGB mode (in case it's grayscale)
    resized_image = resized_image.convert('RGB')
    # Convert the image to a numpy array
    img_array = np.array(resized_image)
    # Normalize the image
    normalized_image = img_array / 255.0
    # Add batch dimension
    processed_image = np.expand_dims(normalized_image, axis=0)
    return processed_image

# Function to make CT scan prediction
def predict_ct(image):
    # Preprocess the image
    processed_image = preprocess_ct_image(image)
    # Perform prediction
    prediction = ct_model.predict(processed_image)
    # Process prediction result
    class_labels = ['Bengin', 'Malignant', 'Normal']  # Update with your class labels
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    # Return prediction
    return predicted_class

# Route for rendering the CT scan page
@app.route('/ct')
def ct():
    return render_template('ct.html')

# Route for handling CT scan predictions
@app.route('/predict_ct', methods=['POST'])
def predict_ct_route():
    # Get the uploaded image
    uploaded_file = request.files['ct_image']
    if uploaded_file.filename != '':
        # Read the image
        img = Image.open(uploaded_file)
        # Perform CT scan prediction
        prediction = predict_ct(img)
        # Return prediction result as JSON
        return jsonify({'result': prediction})
    else:
        return jsonify({'error': 'No file uploaded'})


if __name__ == "__main__":
    app.run(debug=True)
