import ctypes
import numpy as np
import cv2
import random
class SincurlImage(object):

    def __init__(self,probability=0.5):
        self.ll = ctypes.cdll.LoadLibrary
        self.lib = self.ll("./lib/SincurlImgTool.so")
        self.lib.setparams(ctypes.c_float(0.04),ctypes.c_float(0.05))
        self.probability = probability

    def setprob(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        if random.random() > self.probability:
            return img
        img = img.astype(np.uint8)
        dataptr = img.ctypes.data_as(ctypes.c_char_p)
        rows, cols,_ = img.shape  # (h , w, c)
        self.lib.process(dataptr, rows, cols)
        return img