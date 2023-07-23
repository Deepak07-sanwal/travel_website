pip install opencv-python

import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

# Load the .h5 model file
model = load_model('path_to_your_model.h5')

# Function to preprocess input image
def preprocess_image(img_path, target_size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Define the Streamlit app
def main():
    st.title("ML Model Prediction App")

    # Add a file input widget for the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the input image
        input_image = preprocess_image(uploaded_file, (150, 150))

        # Make prediction using the model
        prediction = model.predict(input_image)

        # The 'prediction' variable will contain the predicted outputs for the input image.
        # You can interpret the predictions based on your model's output format and problem.
        # For example, if you have binary classification (e.g., PNEUMONIA vs. NORMAL):
        if prediction[0][0] > 0.5:
            result = 'PNEUMONIA'
        else:
            result = 'NORMAL'

        # Display the result
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("Prediction:", result)

if __name__ == '__main__':
    main()

