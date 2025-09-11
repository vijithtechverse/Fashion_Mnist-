# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("fashion_mnist_cnn.keras")

# Label map
label_names = {
    0: "T-shirt/top", 
    1: "Trouser", 
    2: "Pullover", 
    3: "Dress",
    4: "Coat", 
    5: "Sandal", 
    6: "Shirt", 
    7: "Sneaker",
    8: "Bag", 
    9: "Ankle boot"
}

st.title("👕 Fashion-MNIST Image Classifier")
st.write("Upload an image (JPG, JPEG, PNG) of clothing, resized to **28x28 grayscale**, to classify it.")

# Upload image
uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_img:
    # Open and preprocess image
    img = Image.open(uploaded_img).convert("L")   # Convert to grayscale
    img_resized = img.resize((28, 28))            # Resize to 28x28
    pixels = np.array(img_resized).reshape(28, 28, 1).astype("float32") / 255.0
    
    # Show uploaded image
    st.image(img, caption="Uploaded Image", use_container_width=False, width=150)
    
    # Predict
    pred_probs = model.predict(np.expand_dims(pixels, axis=0))
    pred_class = np.argmax(pred_probs, axis=1)[0]
    
    # Show prediction
    st.success(f"Predicted Class: **{label_names[pred_class]}**")
    
    # Show confidence scores
    st.write("### Confidence Scores:")
    for i, prob in enumerate(pred_probs[0]):
        st.write(f"{label_names[i]}: {prob:.4f}")

