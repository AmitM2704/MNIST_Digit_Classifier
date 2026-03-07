import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model("mnist_model.h5")

uploaded = st.file_uploader("Upload digit")

if uploaded:
    img = Image.open(uploaded).convert('L')
    img = np.array(img)

    img = cv2.resize(img,(28,28))
    img = 255 - img
    img = img/255.0
    img = img.reshape(1,28,28)

    pred = model.predict(img)

    st.write("Prediction:", np.argmax(pred))