import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import numpy as np

# Cache the model loading function
@st.cache_resource
# Load the saved model
def load_model():
    MODEL_PATH = r"C:\Users\Maddy\Documents\Python\Data Science\Deep Learning\Potato_Disease\training\model_v1.keras"
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load the model
model = load_model()


# In[10]:


# Define class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

IMAGE_SIZE = 256

def predict(model, img):
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Streamlit app
st.title("Potato Disease Classification")

# Option to choose between uploading an image or taking a picture
option = st.radio("Select input method:", ('Upload Image', 'Take Picture'))

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make prediction
        predicted_class, confidence = predict(model, image)
        
        # Display results
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")

elif option == 'Take Picture':
    picture = st.camera_input("Take a picture")
    if picture is not None:
        image = Image.open(picture).convert('RGB')
        st.image(image, caption='Captured Image', use_column_width=True)
        
        # Make prediction
        predicted_class, confidence = predict(model, image)
        
        # Display results
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")





