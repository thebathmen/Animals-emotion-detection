import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('animal_emotion_model.h5')

# Define animal and emotion labels
animal_labels = ['dog', 'cat', 'bird', 'rat']
emotion_labels = ['happy', 'sad', 'hungry', 'angry']

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input shape
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Function to make predictions for a single animal
def predict_emotion(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    max_index = np.argmax(prediction)
    animal_label = animal_labels[max_index % len(animal_labels)]  # Adjusting to ensure proper label matching
    emotion_label = emotion_labels[max_index % len(emotion_labels)]  # Adjusting to ensure proper label matching
    return animal_label, emotion_label

# Streamlit app interface
st.title("Animal Emotion Detection")

# Option to select single or multiple images
option = st.selectbox("Choose an option", ("Single Animal Image", "Group of Animal Images"))

if option == "Single Animal Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        animal_category, emotion = predict_emotion(image)
        st.write(f"Predicted Animal: {animal_category}")
        st.write(f"Predicted Emotion: {emotion}")

elif option == "Group of Animal Images":
    uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files is not None:
        emotions = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            animal_category, emotion = predict_emotion(image)
            emotions.append((animal_category, emotion))
            st.image(image, caption=f"{animal_category}: {emotion}", use_column_width=True)
        if emotions:
            st.write("")
            st.write("Classified emotions for the group of images:")
            for animal_category, emotion in emotions:
                st.write(f"{animal_category}: {emotion}")

# Run the Streamlit app
# In your terminal, use the command: streamlit run app.py
