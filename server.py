#!/usr/local/lib/python3.7/
# import barreader
import cv2
from flask import Flask
from flask import request
import json
import numpy as np
import PaddleOCR.tools.infer.utility as utility
import PaddleOCR.tools.infer.predict_system as pocr
import deskew
import subprocess
import os
import io
import pyheif
from PIL import Image

app = Flask(__name__)

@app.route('/bar', methods=['POST'])
def scanBar():
    # Gets the image sent by the client
    img_file = request.files['img'].read()
    # Processes said image
    img = preprocess(img_file)
    barreader.read(img)
    # Repackages results
    dictobj = {'numBlocks': 3, 'texts': [], 'scores': [], 'blocks': []}
    jsonobj = json.dumps(dictobj)
    print('******')
    return jsonobj

@app.route('/', methods=['POST'])
def scan():
    # Gets the image sent by the client
    print('Received request')
    filein = request.files['img']
    # Processes said image
    args = utility.parse_args()
    # args.det_model_dir = "./PaddleOCR/inference/det_sast_tt/"
    # args.det_sast_polygon = True
    # args.det_algorithm = "SAST"
    img = convertcv2(filein)
    img = preprocess(img)
    print("processing")
    dt_boxes, rec_res = pocr.process(args, img, filein.filename)
    # Repackages results
    print("packaging")
    dictobj = {'numBlocks': len(rec_res), 'texts': [], 'scores': [], 'blocks': []}
    # Reformats the texts and scores into separate lists
    for text, score in rec_res:
        dictobj['texts'].append(text)
        dictobj['scores'].append(float(score))
    # Reformats the texts' coordinates
    for box in dt_boxes:
        corners = []
        for pair in box:
            xy = []
            for val in pair:
                xy.append(int(val))
            corners.append(xy)
        dictobj['blocks'].append(corners)
    jsonobj = json.dumps(dictobj)
    print('******')
    return jsonobj


def convertcv2(filein):
    img_name = filein.filename
    img_file = filein.read()
    print("Got image: " + img_name)
    # Converts filein to a cv2 image
    if 'heic' in img_name:
        heifimg = pyheif.read_heif(img_file)
        pilimg = Image.frombytes(heifimg.mode, heifimg.size, heifimg.data, "raw", 
        heifimg.mode, heifimg.stride)
        img = cv2.cvtColor(np.asarray(pilimg), cv2.COLOR_RGB2BGR)
        filein.filename = filein.filename +'.jpg'
    else:
        # Converts string data from client to numpy array
        npimg = np.fromstring(img_file, np.uint8)
        # Converts numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    return img

def preprocess(img):
    print("preprocessing")
    # Deskews image
    img = deskew.correct(img)
    # blurs image 
    # filter = 1/9 * np.array([[1,1,1],[1,1,1],[1,1,1]])
    # blurimg = cv2.filter2D(rotateimg,-1, filter)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img

if __name__ == '__main__':
    app.run()
