import os
import re
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import pickle
import lungs_finder as lf
import imutils
import io
import cv2
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,session,send_file,jsonify
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = pickle.load(open('Final_Covid_Model.pkl','rb'))

app.config['SECRET_KEY'] = 'b7d77a9618fcfdb7ec74f62aac65b0ec'
    
def preprocess(img):
   img1 = cv2.resize(img, (150,150))
   train_imgs = img_to_array(img1)
   train_imgs = np.expand_dims(train_imgs,axis=0)
   train_imgs = preprocess_input(train_imgs)
   train_imgs = train_imgs.reshape(1,150*150*3) 
   return train_imgs


def proportional_resize(image, max_side):
    if image.shape[0] > max_side or image.shape[1] > max_side:
        if image.shape[0] > image.shape[1]:
            height = max_side
            width = int(height / image.shape[0] * image.shape[1])
        else:
            width = max_side
            height = int(width / image.shape[1] * image.shape[0])
    else:
        height = image.shape[0]
        width = image.shape[1]

    return cv2.resize(image, (width, height))
    
    
def lungsfinder(img):
    image = cv2.equalizeHist(img)
    image = cv2.medianBlur(img, 3)
    scaled_image = proportional_resize(image, 512)
    left_lung_haar_rectangle = lf.find_left_lung_haar(scaled_image)
    right_lung_haar_rectangle = lf.find_right_lung_haar(scaled_image)
    if left_lung_haar_rectangle is not None and right_lung_haar_rectangle is not None:
        return str(1)
    else:
        return str(0)


@app.route("/",methods=['POST'])
def predict():
    python_obj = request.get_json(force=True)
    id_ = python_obj["id"]
    file_path = python_obj["image_url"]
    process = python_obj["process_status"]
    prediction_type = python_obj["prediction_type"]
    r = requests.get(file_path)
    dataBytesIO=io.BytesIO(r.content)
    dataBytesIO.seek(0)
    image1 = Image.open(dataBytesIO)
    opencvImage1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
    gray = np.array(opencvImage1, dtype='uint8')
    opencvImage = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    lungs = lungsfinder(opencvImage)
    d = preprocess(opencvImage1)
    model = pickle.load(open('Final_Covid_Model.pkl','rb'))
    pred = model.predict(d)
    pred_class1 = np.argmax(pred,axis=-1)


    if prediction_type == 1:
        if lungs == '1':
            if pred_class1 == [1]:
                prediction = 'The Patient has Covid Signatures'
                process_status = 1
                return jsonify(id=id_,prediction_result=prediction,process_status=process_status,prediction_type=1,prediction_status=1,lungs=1)
            elif pred_class1 == [2]:
                prediction = 'The Patient has Tuberculosis Signatures'
                process_status = 1
                return jsonify(id=id_,prediction_result=prediction,process_status=process_status,prediction_type=1,prediction_status=1,lungs=1)
            elif pred_class1 == [3]:
                prediction = 'The Patient has Viral Pneumonia Signatures'
                process_status = 1
                return jsonify(id=id_,prediction_result=prediction,process_status=process_status,prediction_type=1,prediction_status=1,lungs=1)
            else:
                prediction = 'The Patient has Normal Signatures'
                process_status = 1
                return jsonify(id=id_,prediction_result=prediction,process_status=process_status,prediction_type=1,prediction_status=0,lungs=1)
        else:    
            return jsonify(id=id_,prediction_result='Please Import Only Lung Image',process_status=0,prediction_type=1,prediction_status='',lungs=0)

if __name__ == '__main__':
    app.run(debug=True)
