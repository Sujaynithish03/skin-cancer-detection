import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st

# Load the pre-trained model
model = load_model('skin_lesion_model.h5')

# Define the lesion type mapping
lesion_type_dict = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses or Skin going to be shredded soon',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

# Function to preprocess the image for model input
def preprocess_image(image, size):
    image = np.array(image)
    img = cv2.resize(image, (size, size))  # Resize to 128x128
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to classify the image using the model
def classify_image(model, preprocessed_image):
    predictions = model.predict(preprocessed_image)
    predicted_class_idx = np.argmax(predictions, axis=1)
    predicted_class = lesion_type_dict[predicted_class_idx[0]]
    return predicted_class

# Page layout enhancements
st.set_page_config(
    page_title="Skin Lesion Classification",
    layout="wide",  # Full-width layout
    initial_sidebar_state="expanded"
)

# Main UI Title
st.title("🩺 Skin Lesion Classification")
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 30px;
        color: #3D5B99;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .markdown-text {
        font-size: 18px;
        color: #444;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    <div class="title">Welcome to the Skin Lesion Classifier</div>
    <div class="markdown-text">
        Upload an image of a skin lesion, and the AI model will classify it.<br>
        For any concerns, please seek professional medical advice.
    </div>
    """, unsafe_allow_html=True
)

# Create a two-column layout
col1, col2 = st.columns(2)

# Allow user to upload an image in the first column
with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose a skin lesion image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to an image and display it
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image, 128)

        # Classify the image
        with st.spinner("Classifying..."):
            predicted_class = classify_image(model, preprocessed_image)

        # Display the prediction in the second column
        with col2:
            st.header("Prediction")
            if predicted_class == lesion_type_dict[4]:
                st.success("Normal (Actinic keratoses or Skin going to be shredded soon)", icon="✅")
            else:
                st.success(f"Predicted Class: {predicted_class}", icon="✅")

            # Additional note
            st.markdown(
                """
                <style>
                .important-note {
                    color: #FF5733;
                    font-weight: bold;
                }
                </style>
                <div class="important-note">
                    **Important Note:** This application is for educational purposes only and is not a substitute for professional medical diagnosis.
                </div>
                """, unsafe_allow_html=True
            )
    else:
        st.info("Please upload an image to classify.")

# Sidebar for additional information
st.sidebar.header("About this App")
st.sidebar.markdown(
    """
    This app uses a pre-trained deep learning model to classify skin lesions into different categories.
    It helps raise awareness but should not replace medical diagnosis.
    """
)
st.sidebar.info("Created using Streamlit and TensorFlow")

# Footer
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    .main-footer {font-size: 12px; text-align: center; color: #999;}
    </style>
    <div class="main-footer">© 2024 Skin Lesion AI Classifier - All Rights Reserved</div>
    """, unsafe_allow_html=True
)
