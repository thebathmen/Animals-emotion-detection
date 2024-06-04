import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps

# Define animal and emotion labels
animal_labels = ['dog', 'cat', 'bird', 'rat', 'unknown_animal']
emotion_labels = ['happy', 'sad', 'hungry', 'unknown_emotion']

# Load the trained model
model = tf.keras.models.load_model('C:\\Users\\Addmin\\Desktop\\pets emotion detection\\animal_emotion_model.h5')

# Function to preprocess image
def preprocess_image(image):
    size = (128, 128)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image = np.asarray(image)
    image = image / 255.0
    return image

# Function to make predictions for a single image
def predict_emotion(image):
    image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(image, axis=0))
    max_index = np.argmax(prediction)
    animal_label = animal_labels[max_index] if max_index < len(animal_labels) else 'unknown_animal'
    emotion_label = emotion_labels[max_index] if max_index < len(emotion_labels) else 'unknown_emotion'
    return animal_label, emotion_label

st.title('Animal Emotion Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    animal_category, emotion = predict_emotion(image)
    st.write(f"Predicted Animal: {animal_category}")
    st.write(f"Predicted Emotion: {emotion}")
