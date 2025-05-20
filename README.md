# Dog Breed Identifier Web Application

This project is a simple web application that allows users to upload an image of a dog, and then uses a pre-trained deep learning model to predict the dog's breed and provide a confidence score.

## Features

* **Image Upload:** Easily upload dog images via drag-and-drop or file selection.
* **Breed Prediction:** Utilizes a TensorFlow/Keras model to predict the dog's breed from a large dataset.
* **Confidence Score:** Displays the prediction confidence for the identified breed.
* **User-Friendly Frontend:** A clean and responsive user interface built with HTML, CSS (Tailwind CSS), and JavaScript.
* **Python Backend:** A Flask server handles image processing and model inference.

## Technologies Used

* **Backend:**
    * Python 3.x
    * Flask (Web Framework)
    * TensorFlow / Keras (Deep Learning Library)
    * Pillow (PIL - Image processing)
    * Numpy (Numerical operations)
    * Requests (for potential external API calls, e.g., Dog API for sample images if that feature was active)
    * Flask-CORS (for handling Cross-Origin Resource Sharing during development)
* **Frontend:**
    * HTML5
    * Tailwind CSS (for styling)
    * JavaScript (for dynamic interactions and API calls)
* **Model:**
    * Pre-trained Convolutional Neural Network (`dog_breed_model.h5`), likely based on **MobileNetV2** architecture for efficient inference.

## Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository (if you haven't already)

If you're setting this up on a new machine, clone your project from GitHub:

```bash
git clone [https://github.com/subrahmanyam024/dog-breed-identifier.git](https://github.com/subrahmanyam024/dog-breed-identifier.git)
cd dog-breed-identifier

Replace subrahmanyam024/dog-breed-identifier.git with your actual GitHub repository URL.

2. Backend Setup
It's highly recommended to use a Python virtual environment to manage dependencies.

Bash

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required Python packages
pip install flask tensorflow pillow numpy requests flask-cors
3. Model File
Ensure you have the dog_breed_model.h5 file in the root directory of your project. This is the trained AI model that the backend uses for predictions.
Note: If your model file is very large (e.g., >100MB) and was excluded from Git using .gitignore, you will need to place this file manually into your project directory after cloning/setting up.
4. Frontend Files
Ensure you have index.html, script.js, and style.css (if you have one) in the root directory of your project.
Running the Application
Start the Flask Backend:
Make sure your virtual environment is active (from Backend Setup step 2).

Bash

python app.py
The backend will start, usually running on http://127.0.0.1:5000/. You should see messages in your terminal indicating the Flask server has started and the model has been loaded.

Open the Frontend:
Open your web browser and navigate to the index.html file directly. You can typically do this by:

Opening your file explorer/finder, finding index.html in your project folder, and double-clicking it.
Or, in your browser's address bar, type file:///path/to/your/project/index.html (replace with the actual path).
The web interface will appear, and you can now upload dog images for prediction.

Usage
Open the index.html file in your web browser.
Click the "Upload Image" button or drag and drop an image file into the designated area.
The application will display the uploaded image, show a loading spinner, and then present the predicted dog breed and confidence score.
Future Enhancements (Potential Ideas)
Display Top N Predictions: Show the top 3 or 5 most likely breeds instead of just the highest one.
Improve Model Accuracy: Fine-tune the model with more diverse or specific dog breed datasets.
Image Preprocessing UI: Allow users to crop or rotate images before prediction.
Deployment: Deploy the application to a cloud platform (e.g., Heroku, Google Cloud, AWS) to make it accessible online.
More Robust Error Handling: Provide more user-friendly messages for various backend errors.
