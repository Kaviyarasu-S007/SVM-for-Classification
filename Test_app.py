import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Streamlit app
st.title("Image Classification App")

# Input image link
image_link = st.text_input("Enter Image Link:")

if image_link:
    try:
        # Fetch the image from the link
        response = requests.get(image_link)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Preprocess the image
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Predict the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Display the image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Display classification results
        st.subheader("Classification:")
        st.write(f"Class: {class_name}")
        st.write(f"Confidence Score: {confidence_score * 100:.2f}%")

        # Display progress bars for each class
        st.subheader("Confidence Scores:")
        st.text("Dog")
        progress_bar_class_1 = st.progress(int(prediction[0][0] * 100))  
        st.text("Cat")
        progress_bar_class_2 = st.progress(int(prediction[0][1] * 100))  

        st.text(f"Dog: {class_names[0].strip()} - {prediction[0][0] * 100:.2f}%")
        st.text(f"Cat: {class_names[1].strip()} - {prediction[0][1] * 100:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")
