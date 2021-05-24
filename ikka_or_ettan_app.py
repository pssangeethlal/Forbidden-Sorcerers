import streamlit as st

from keras.models import load_model
st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def load_model1():
    model = load_model('best_modelA_2.h5')
    return model
model=load_model1()
st.write("""
        # The Mohanlal Mammootty Classifier!!!
         """)
file = st.file_uploader("Please upload the Image", type=["jpg","jpeg","png"])
import numpy as np
from keras.preprocessing import image
from PIL import Image, ImageOps
def import_and_predict(test_image, model):
    size = (227,227)
    test_image = ImageOps.fit(test_image, size, Image.ANTIALIAS)

    test_image = image.img_to_array(test_image)
    test_image = np.array([test_image], dtype=np.float16) / 255.0
    prediction = model.predict(test_image)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    test_image = Image.open(file)
    st.image(test_image, width= 400)
    predictions = import_and_predict(test_image,model)
    
    categories = ["Mammootty","Mohanlal"]
    if np.max(predictions)  >= 0.8:
        string="The Uploaded image most likely to be "+categories[np.argmax(predictions)]
    else:
        string="Oops! we don't know Who or what is this"
    st.success(string)
    
    
    
