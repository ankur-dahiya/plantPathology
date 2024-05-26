import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow_hub as hub
import matplotlib.pyplot as plt

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('Plant Pathology')

@st.cache_resource
def load_trained_model():
    model_path = r'apple3.h5'
    classifier_model = load_model(model_path, compile=False)
    return classifier_model

model = load_trained_model()

def main():
    file_uploaded = st.file_uploader('Choose an image of an Apple Leaf - disease will be Predicted', type='jpg')
    if file_uploaded is not None:
        img = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(img)
        plt.axis('off')
        st.pyplot(figure)
        result = predict_class(img)
        st.write(f'Prediction: {result}')

def predict_class(img):
    shape = ((224, 224, 3))
    test_image = img.resize((224, 224))
    test_image = keras_image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image)
    categories = ['Healthy', 'Multiple Disease', 'Rust', 'Scab']
    pred_class = result.argmax()
    output = categories[pred_class]
    return output

footer = """<style>
a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
}

a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: black;
    text-align: center;
}
</style>

<div class="footer">
<p align="center"> <a href="#">Developed by Ankur & Ronak Singh</a></p>
</div>
        """

st.markdown(footer, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
