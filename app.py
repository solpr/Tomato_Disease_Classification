import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

class_names = ['Tomato_Bacterial_spot',
                'Tomato_Early_blight',
                'Tomato_Late_blight',
                'Tomato_Leaf_Mold',
                'Tomato_Septoria_leaf_spot',
                'Tomato_Spider_mites_Two_spotted_spider_mite',
                'Tomato__Target_Spot',
                'Tomato__Tomato_YellowLeaf__Curl_Virus',
                'Tomato__Tomato_mosaic_virus',
                'Tomato_healthy']


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model_2.hdf5')
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("# Tomato Disease Classification ... prepared by Solomon Araya")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def predict(model, img):
    img = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    class_name, confidence = predict(model, image)
    output = f"this image most likely belongs to {class_name} with a {confidence}% confidence"
    st.write(output)
