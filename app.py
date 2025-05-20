from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import tensorflow_datasets as tfds
import requests # New import for making HTTP requests

app = Flask(__name__)
CORS(app) # Enable CORS for local development

# --- Global variables for model and class names ---
trained_model = None
CLASS_NAMES = None
IMG_SIZE = 224 # Matches the size used during training
CONFIDENCE_THRESHOLD = 0.5 # Your determined threshold

# --- Helper function to get Dog API friendly breed name ---
# Converts Stanford name (e.g., 'n02100877-irish_setter') to Dog API format (e.g., 'irishsetter')
def _get_dog_api_breed_name(stanford_breed_name):
    try:
        # Extract the "breed_name" part after 'nXXXXXXXXX-'
        parts = stanford_breed_name.split('-')
        if len(parts) > 1:
            breed_part = parts[1]
        else: # Fallback if name format is unexpected
            breed_part = stanford_breed_name

        # Convert to lowercase and remove underscores for API
        api_name = breed_part.replace('_', '').lower()

        # Handle specific exceptions for Dog API if needed (rare, but possible)
        # For example, if "st.bernard" was expected instead of "saintbernard"
        # if api_name == "stbernard":
        #     return "st.bernard" # This is just an example

        return api_name
    except Exception as e:
        print(f"Error processing breed name for Dog API: {e}")
        return None # Indicate failure

# --- Helper function to fetch a random image for a breed from Dog API ---
def get_breed_image_url(breed_api_name):
    if not breed_api_name:
        return None
    try:
        # Construct API URL for a random image of a specific breed
        response = requests.get(f"https://dog.ceo/api/breed/{breed_api_name}/images/random")
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()
        if data['status'] == 'success':
            return data['message'] # This is the image URL
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from Dog API for {breed_api_name}: {e}")
    except Exception as e:
        print(f"Error parsing Dog API response for {breed_api_name}: {e}")
    return None


# --- Model Loading and Breed Name Initialization ---
def load_resources():
    global trained_model, CLASS_NAMES

    # Load the trained model
    try:
        trained_model = tf.keras.models.load_model('dog_breed_model.h5')
        print("Trained dog_breed_model.h5 loaded successfully.")
    except Exception as e:
        print(f"Error loading trained model: {e}")
        trained_model = None

    # Load class names from TensorFlow Datasets
    try:
        # We only need the info, so we can load it without the full dataset
        _, info = tfds.load('stanford_dogs', split='train', with_info=True)
        CLASS_NAMES = info.features['label'].names
        print("Dog breed class names loaded successfully.")
    except Exception as e:
        print(f"Error loading dog breed class names: {e}")
        CLASS_NAMES = [f"Breed {i}" for i in range(120)] # Fallback if loading fails

# Call load_resources when the app starts
with app.app_context():
    load_resources()

@app.route('/', methods=['GET'])
def index():
    return "Dog Breed Identifier Backend is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if trained_model is None:
        return jsonify({'error': 'AI model not loaded on server. Please check server logs.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Read and preprocess the image
        img = Image.open(image_file).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array) # Preprocess for MobileNetV2

        # Make a prediction using the trained model
        predictions = trained_model.predict(processed_img)
        predicted_class_id = np.argmax(predictions)
        confidence = np.max(predictions)

        predicted_breed = "Unknown" # Default display string
        breed_image_url = None # Default no image URL

        # Apply confidence threshold for "Not a Dog"
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_breed = "Not a Dog (or unclear image)"
            confidence = 0.0 # Reset confidence to 0 if classified as not a dog
        else:
            # Get the Stanford breed name (e.g., 'n02100877-irish_setter')
            predicted_breed_stanford_name = CLASS_NAMES[predicted_class_id]
            # Format for display (e.g., "Irish Setter")
            predicted_breed = predicted_breed_stanford_name.split('-')[1].replace('_', ' ').title()

            # Get image URL from Dog API
            dog_api_breed_name = _get_dog_api_breed_name(predicted_breed_stanford_name)
            if dog_api_breed_name: # Only fetch if conversion was successful
                breed_image_url = get_breed_image_url(dog_api_breed_name)


        return jsonify({
            'breed': predicted_breed,
            'confidence': float(confidence),
            'breed_image_url': breed_image_url # Include the image URL in the response
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f"Error processing image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)