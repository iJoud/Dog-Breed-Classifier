
from flask import Flask
from flask import render_template, request

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

import numpy as np
from glob import glob
import cv2 
import json
import os


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def extract_InceptionV3(tensor):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    # extract bottleneck features from the model
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def predict_dog_breed(image_path):
    # 1. Extract the bottleneck features corresponding to the chosen CNN model
    bottleneck_feature = extract_InceptionV3(path_to_tensor(image_path))
    # 2. Supply the bottleneck features as input to the model to return the predicted vector
    prediction = Inception_model.predict(bottleneck_feature)
    # 3. Use the dog_names array to return the corresponding breed.
    dog_breed = dog_names[np.argmax(prediction)]

    return dog_breed.split('.')[-1]

def detection_algorithm(image_path):

    prediction = ''

    if face_detector(image_path):
        prediction+='The image is for a Human!<br>'
        prediction+='The resembling dog breed is: '+ predict_dog_breed(image_path)
    elif dog_detector(image_path):
        prediction+='The image is for a dog!<br>'
        prediction+='The dog breed is: '+ predict_dog_breed(image_path)
    else:
        prediction+='Error Occurred!<br>'
        prediction+='The provided image is neither for a human or a dog'

    return prediction


# read all dogs names
with open("dog_names", "r") as f:
    dog_names = json.load(f)

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    '../project/haarcascades/haarcascade_frontalface_alt.xml')

# define ResNet50 model for dog detections 
ResNet50_model = ResNet50(weights='imagenet')

# define my model architucture
Inception_model = Sequential()
Inception_model.add(GlobalAveragePooling2D(input_shape=(5, 5, 2048)))
Inception_model.add(Dense(133, activation='softmax'))

# load my fine-tuned model weights
Inception_model.load_weights(
    '../project/saved_models/weights.best.Inception.hdf5')

app = Flask(__name__, static_url_path='/static')

# web application home page
@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    return render_template('index.html')


# classify the uploaded image dog breed
@app.route('/prediction', methods = ['POST'])
def prediction():
    # delete all images saved previously in the folder
    for f in glob('./static/*'):
        print(f)
        os.remove(f)

    # save the image in the folder
    f = request.files['uploaded_image']
    img_path = 'static/'+f.filename
    f.save(img_path)
    
    # create dictionary for predictions
    pred = {
        'prediction':detection_algorithm(img_path),
        'uploaded_image': img_path
    }
    
    # send the dictionary in json format
    return json.dumps(pred)


def main():
    app.run(host='0.0.0.0', port=8000, debug=True)


if __name__ == '__main__':
    main()
