import tensorflow as tf
import keras

from PIL import Image
import os, glob
import numpy as np

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from keras.models import load_model

import ctypes
import _ctypes
import cv2
import sys

import os, glob

'''Google MobileNet model for Keras.
# Reference:
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
'''

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

class InfraRedRuntime(object):
    def __init__(self):

        # Loop until the user clicks the close button.
        self._done = False

        # Kinect runtime object, we want only color and body frames
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared)

        # here we will store skeleton data
        self._bodies = None

        # Load Keras model
        self.model=load_model('model2.h5')


    def face_recognition(self,img):
        img2 = cv2.resize(img, (224, 224))
        data = img2.reshape(1, 224, 224, 3)
        label = self.model.predict(data)
        index = np.argmax(label)
        if index == 0:
            str1 = "Human"
            cv2.putText(img, str1, (100, 100), 2, 1, (0, 255, 0))
        else:
            str1 = "Nega"
            cv2.putText(img, str1, (100, 100), 2, 1, (0, 255, 0))
        cv2.imshow("Infrared",img)
        cv2.waitKey(3)

    def run(self):
        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            # --- Getting frames and drawing
            if self._kinect.has_new_infrared_frame():
                frame = self._kinect.get_last_infrared_frame()
                width = self._kinect.infrared_frame_desc.Width
                height = self._kinect.infrared_frame_desc.Height
                f8 = np.uint8(frame.clip(1, 25000) / 128.)
                img8b = f8.reshape((height, width))
                img = np.dstack((img8b, img8b, img8b))
                self.face_recognition(img)
                frame = None

            # --- Wait to exit: Press ESC button
            c = cv2.waitKey(20)
            if c == 27:
                self._done = True

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()


if __name__ == '__main__':
    game = InfraRedRuntime();
    game.run();