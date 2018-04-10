#!/usr/bin/env python3

import os

import time
import datetime
import cv2
import numpy as np
import uuid
import json

import functools
import logging
import collections
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

### the webserver
from flask import Flask, request, render_template
import argparse



import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import xml.dom.minidom
# %matplotlib inline
from nms import nms
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = './'  # this file is expected to be in {caffe_root}/examples
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
#caffe.set_device(0)
caffe.set_mode_cpu()

model_def = './examples/TextBoxes/deploy.prototxt'
model_weights = './examples/TextBoxes/TextBoxes_icdar13.caffemodel'

use_multi_scale = False

if not use_multi_scale:
	scales=((700,700),)
else:
	scales=((300,300),(700,700),(700,500),(700,300),(1600,1600))


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)





def run(inpath, outpath):
    dt_results=[]
    image=caffe.io.load_image(inpath)
    image_height,image_width,channels=image.shape
    plt.clf()
    plt.imshow(image)
    currentAxis = plt.gca()

    img = cv2.imread(inpath)
    for scale in scales:
        print(scale)
        image_resize_height = scale[0]
        image_resize_width = scale[1]
        transformer = caffe.io.Transformer({'data': (1,3,image_resize_height,image_resize_width)})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104,117,123])) # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        
        net.blobs['data'].reshape(1,3,image_resize_height,image_resize_width)		
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        # Forward pass.
        detections = net.forward()['detection_out']
        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
        top_conf = det_conf[top_indices]
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            xmin = max(1,xmin)
            ymin = max(1,ymin)
            xmax = min(image.shape[1]-1, xmax)
            ymax = min(image.shape[0]-1, ymax)
            score = top_conf[i]
            dt_result=[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax,score]
            dt_results.append(dt_result)
            
    dt_results = sorted(dt_results, key=lambda x:-float(x[8])) 
    nms_flag = nms(dt_results,0.3)

    for k,dt in enumerate(dt_results):
        if nms_flag[k]:
            name = '%.2f'%(dt[8])
            xmin = int(dt[0])
            ymin = int(dt[1])
            xmax = int(dt[2])
            ymax = int(dt[5])
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = 'b'
            crop_img = img[ymin:ymax, xmin:xmax]
            txt = pytesseract.image_to_string(crop_img)
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, txt, bbox={'facecolor':'white', 'alpha':0.5})

    plt.savefig(outpath)






class Config:
    SAVE_DIR = 'examples/TextBoxes/static/results'


config = Config()


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')



@app.route('/', methods=['POST'])
def index_post():
    global predictor
    import io

    session_id = str(uuid.uuid1())

    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)

    dirpath = os.path.join(config.SAVE_DIR, session_id)
    os.makedirs(dirpath)

    # save input image
    in_path = os.path.join(dirpath, 'input.png')
    cv2.imwrite(in_path, img)

    # save illustration
    output_path = os.path.join(dirpath, 'output.png')
    run(in_path, output_path)
    return render_template('index.html', session_id=session_id)


def main():
    global checkpoint_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=1234, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    app.debug = True
    app.run('0.0.0.0', args.port)

if __name__ == '__main__':
    main()

