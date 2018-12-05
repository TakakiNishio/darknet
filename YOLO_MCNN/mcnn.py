#python library
# import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *
import time

#openCV
import cv2

#chainer library
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import serializers
import network_structure as nn


# load MCNN classifier
class MCNN:

    def __init__(self, model_path, model, image_size):

        model_name = 'cnn_gpu.model'
        self.size = image_size

        print ("Loading MCNN classifier model...")
        serializers.load_npz(model_path+model_name, model)
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        chainer.cuda.get_device(0).use()
        model.to_gpu()

        self.model = model

    def __call__(self, img):

        img = cv2.resize(img, (self.size,self.size), interpolation = cv2.INTER_LINEAR)
        img = img.reshape((1,3,self.size,self.size)).astype(np.float32) / 255.
        validation_output = self.model.forward(chainer.Variable(cuda.cupy.array(img))).data
        prob = F.softmax(validation_output).data[0][1]
        label = np.argmax(validation_output)

        return label, prob

class MCNN_grayed:

    def __init__(self, model_path, model, image_size):

        model_name = 'cnn_gpu.model'
        self.size = image_size

        print ("Loading MCNN classifier model...")
        serializers.load_npz(model_path+model_name, model)
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        chainer.cuda.get_device(0).use()
        model.to_gpu()

        self.model = model

    def __call__(self, img):

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.size,self.size), interpolation = cv2.INTER_LINEAR)
        img = img.reshape((1,1,self.size,self.size)).astype(np.float32) / 255.
        validation_output = self.model.forward(chainer.Variable(cuda.cupy.array(img))).data
        prob = F.softmax(validation_output).data[0][1]
        label = np.argmax(validation_output)

        return label, prob

