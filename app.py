import streamlit as st
from PIL import Image
import numpy as np
import joblib

@st.cache_resource
def load_model():
    try:
        model = joblib.load('random_forest_fashion_mnist.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'random_forest_fashion_mnist.pkl' not found. Please upload it.")
        return None

model = load_model()
if model is None:
    st.stop()

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

st.title("Fashion MNIST Image Classifier")
st.write("Upload a fashion image to classify.")

uploaded_file = st.file_uploader("Upload image", type=["png","jpg","jpeg","webp"])
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("L").resize((28,28))
        img_arr = np.array(img) / 255.0
        img_flat = img_arr.flatten().reshape(1, -1)
        st.image(img, caption='Uploaded Image (gray 28x28)', width=150)

        pred = model.predict(img_flat)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(img_flat)[0])

        st.markdown(f"**Prediction:** {class_names[pred]}")
        if confidence is not None:
            st.markdown(f"**Confidence:** {confidence:.2f}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload an image file.")
