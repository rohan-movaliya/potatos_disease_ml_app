import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
# from tensorflow import keras
import tensorflow_hub as hub

st.set_page_config(page_title="Potato Leaf Disease Prediction", page_icon="🥔")

hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Apply background styles
with open('background.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .title-text {
        font-size: 48px;
        font-weight: bold;
        color: #2E4057;  /* Dark blue color */
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader-text {
        font-size: 24px;
        font-weight: bold;
        color: #D98880;  /* Salmon color */
        text-align: center;
        margin-bottom: 30px;
    }
    table {
        color: #ffffff;  /* White color for table text */
        width: 60%;
        margin: 0 auto;
        border-collapse: collapse;
        text-align: center;
    }
    th {
        background-color: #4CAF50;  /* Dark blue color for table header */
        padding: 10px;
        text-align : center;
    }
    td {
        background-color: #2E4057;  /* Dark gray-blue color for table cells */
        padding: 10px;
        font-weight: bold;
        text-align : center;
    }
    .uploaded-image-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("<h1 class='title-text'>Potato Leaf Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subheader-text'>Using Deep Learning to Identify Early Blight, Late Blight, and Healthy Leaves</h3>", unsafe_allow_html=True)


def main():
    file_uploaded = st.file_uploader('Choose an image...', type='jpg')
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        st.image(image, caption='Uploaded Image.', width=300,output_format='JPEG',use_column_width=True)
        result, confidence = predict_class(image)
        st.write("<style>table {color: white;}</style>", unsafe_allow_html=True)
        st.write(
            f"<div style='text-align: center;'>"
            f"<table style='border-collapse: collapse; width: 50%; margin: 0 auto;'>"
            f"<tr style='background-color: #4CAF50;'>"
            f"<th style='text-align: left; padding: 8px;text-align : center;'>Prediction</th>"
            f"<th style='text-align: left; padding: 8px;text-align : center;'>Confidence (%)</th>"
            f"</tr>"
            f"<tr>"
            f"<td style='padding: 8px; font-weight: bold;text-align : center;'>{result}</td>"
            f"<td style='padding: 8px; font-weight: boldtext-align : center;;'>{confidence:.2f}</td>"
            f"</tr>"
            f"</table>"
            f"</div>",
            unsafe_allow_html=True
        )

def predict_class(image):
    try:
        classifier_model = keras.models.load_model('final_model.h5', compile=False)
    except Exception as e:
        st.error("Error loading model. Please make sure 'final_model.h5' exists.")
        return "Error", 0.0

    shape = (256, 256, 3)
    input_image = keras.layers.Input(shape=shape)
    classifier_output = hub.KerasLayer(classifier_model)(input_image)
    new_model = keras.models.Model(inputs=input_image, outputs=classifier_output)

    test_image = image.resize((256, 256))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']

    try:
        prediction = new_model.predict(test_image)
        confidence = round(100 * np.max(prediction[0]), 2)
        final_pred = class_names[np.argmax(prediction)]
        return final_pred, confidence
    except Exception as e:
        st.error("Error predicting. Please try with a different image.")
        return "Error", 0.0

footer = """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #2E4057; /* Matching footer background color */
            color: #ffffff;
            text-align: center;
            padding: 10px 0;
            z-index: 10;
        }
        .footer a {
            color: #ffffff;
            text-decoration: None;
        }
        .footer a:hover {
            color: #ff0000;
        }
        .college-name {
        font-size: 20px;
        font-weight: bold;
        }
    </style>
    <div class="footer">
        <p class="college-name">Silver Oak University - Ahmedabad</p>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
