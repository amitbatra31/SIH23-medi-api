from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess_input
from keras.preprocessing import image
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
app = Flask(_name_)
api = Api(app)
resnet_model = tf.keras.models.load_model('resnet_model.h5')
mobilenet_model = tf.keras.models.load_model('mobilenet_model.h5')
# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')

# Define a function to preprocess an image based on the selected model


def preprocess_image(image_path, model_name):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    if model_name == 'resnet':
        img = resnet_preprocess_input(img)
    elif model_name == 'mobilenet':
        img = mobilenet_preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Define an API endpoint for prediction


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image from the request
        print(request.files.keys())
        uploaded_image = request.files['image']
        print(uploaded_image)
        if uploaded_image:
            image_path = 'temp_image.jpg'
            uploaded_image.save(image_path)

            # Preprocess the image based on the selected model
            model_name = request.form.get('model')
            img = preprocess_image(image_path, model_name)

            # Make predictions
            if model_name == 'resnet':
                prediction = resnet_model.predict(img)
            elif model_name == 'mobilenet':
                prediction = mobilenet_model.predict(img)

            # Get the predicted class label
            print(prediction)
            predicted_class = np.argmax(prediction, axis=1)[0]
            print(predicted_class)
            return jsonify({'class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)})


if _name_ == 'main':
    # flask
    app.run(debug=True, port=3000)