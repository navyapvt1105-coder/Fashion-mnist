import streamlit as st
from PIL import Image
import numpy as np
import joblib

# Load model
model = joblib.load('random_forest_fashion_mnist.pkl')
class_names = [
    'T-shirt/top','Trouser','Pullover','Dress','Coat',
    'Sandal','Shirt','Sneaker','Bag','Ankle boot'
]

st.title("Fashion MNIST Classifier")
st.write(
    """
    Upload a fashion image (square, ideally 28x28 grayscale).
    This application will predict the fashion category using a Random Forest model.
    """)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L").resize((28,28))
    st.image(img, caption="Uploaded image (resized to 28x28)", use_column_width=False)
    arr = np.array(img) / 255.0
    arr = arr.flatten().reshape(1, -1)
    pred = model.predict(arr)[0]
    try:
        confidence = max(model.predict_proba(arr)[0])
    except Exception:
        confidence = None
    st.write(f"**Prediction:** {class_names[pred]}")
    if confidence is not None:
        st.write(f"**Confidence:** {confidence:.2f}")
