# app.py
import streamlit as st
from PIL import Image
import numpy as np
import joblib

# Load the trained model (make sure random_forest_fashion_mnist.pkl is in the same folder)
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('random_forest_fashion_mnist.pkl')
    return model

model = load_model()

# Fashion MNIST class names
class_names = [
    'T-shirt/top','Trouser','Pullover','Dress','Coat',
    'Sandal','Shirt','Sneaker','Bag','Ankle boot'
]

st.title("Fashion MNIST Image Classifier")
st.write("""
Upload a fashion item image (preferably square, any size).  
The image will be resized to 28x28 and converted to grayscale before prediction by the trained Random Forest model.
""")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    try:
        # Open and preprocess the image
        img = Image.open(uploaded_file).convert('L')  # convert to grayscale
        img = img.resize((28, 28))                    # resize to 28x28 pixels
        img_array = np.array(img) / 255.0             # normalize pixels to [0,1]

        # Display the uploaded image
        st.image(img, caption='Uploaded Image (grayscale, resized)', use_column_width=False)

        # Flatten array to fit Random Forest input shape
        img_flat = img_array.flatten().reshape(1, -1)

        # Make prediction
        prediction = model.predict(img_flat)[0]

        # Try to get confidence (probability) if available
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(img_flat)[0]
            confidence = np.max(proba)

        st.markdown(f"**Predicted Category:** {class_names[prediction]}")
        if confidence is not None:
            st.markdown(f"**Confidence:** {confidence:.2f}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload an image file to get a prediction.")
