import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the Keras model
model = load_model('braintumor.h5')

# Function to preprocess the image
def preprocess_image(img):
    # Resize the image to 150x150 and convert to RGB if necessary
    img = cv2.resize(img, (150, 150))
    if len(img.shape) == 2:  # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 1:  # Single-channel image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

# Function to make predictions
def predict(img):
    # Preprocess the image
    processed_img = preprocess_image(img)
    # Expand dimensions to make it 4-dimensional (batch size = 1)
    processed_img = np.expand_dims(processed_img, axis=0)
    # Perform prediction
    predictions = model.predict(processed_img)
    indices = np.argmax(predictions)
    probabilities = np.max(predictions)
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    return labels[indices], float(probabilities)

def main():
    # Set the title of the app
    st.title('Brain Tumor Detection App')

    # Add file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    # Check if a file was uploaded
    if uploaded_file is not None:
        # Read the image file as a NumPy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Make predictions
        label, probability = predict(img)

        # Display the prediction result
        st.write('Prediction:', label)
        st.write('Probability: ',  round(probability*100,2),'%')

# Run the app
if __name__ == '__main__':
    main()
