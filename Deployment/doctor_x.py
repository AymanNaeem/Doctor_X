# Image Processing Packages
from PIL import Image
import base64
import numpy as np
import io
import matplotlib.pyplot as plt

#Deep Learning Packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.preprocessing import image

#Flask API packages
from flask import request
from flask import jsonify
from flask import Flask

#Flask cross origin Technology
from flask_cors import CORS, cross_origin


#Defining Flask app & CORS 
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


#Labels for Diseases to be used in the model training
labels = ['Cardiomegaly', 
      'Emphysema', 
      'Effusion', 
      'Hernia', 
      'Infiltration', 
      'Mass', 
      'Nodule', 
      'Atelectasis',
      'Pneumothorax',
      'Pleural_Thickening', 
      'Pneumonia', 
      'Fibrosis', 
      'Edema', 
      'Consolidation']

# In[4]:

# Using Transfere Learning 
def get_model():
    global model
    # Using pre-trained model from tensorflow library
    base_model = DenseNet121(weights='densenet.hdf5', include_top=False)

    x = base_model.output

    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # add a logistic layer(0,1)
    predictions = Dense(len(labels), activation="sigmoid")(x)
    #Loading Model weights
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights("pretrained_model.h5")
    print(" * Model loaded!")
get_model()

#Load and preprocess image.
def load_image(img, preprocess=True, H=320, W=320):

    if img.mode != "RGB":
        img = img.convert("RGB")
    x = img.resize((H,W))

    if preprocess:
        x = np.array(x)
        mean = x.mean()
        std = x.std()
        x = x-mean
        x = x/std
        x = np.expand_dims(x, axis=0)

    return x    


#API endpoint 
@app.route("/predict", methods=["POST"])
@cross_origin()

def predict():
    labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']

    #getting image from api
    message = request.get_json(force=True)
    #change the image from base64 format to image format
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    
    processed_image = load_image(image)

    prediction = model.predict(processed_image)
    lbl = labels
    preds = {}
    # Assigning each value to it's Disease label
    for value in prediction[0]:
        for key in lbl:
            preds[key] = str(round(value,2))
            lbl.remove(key)
            break
    response = preds
    #Sending the result in JSON format 
    return jsonify(response)

