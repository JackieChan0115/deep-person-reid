import ctypes
import numpy as np

class FloodFill(object):

    def __init__(self):
        self.ll = ctypes.cdll.LoadLibrary
        # self.lib = self.ll("./lib/FloodfillTool.so")
        self.lib = self.ll("./lib/FloodfillTool.so")

    def __call__(self, img:np.ndarray, batchimgs:list ,rands: list)->np.ndarray:
        img = img.astype(np.uint8)
        dataptr = img.ctypes.data_as(ctypes.c_char_p)
        rows, cols,_ = img.shape  # (h , w, c)
        rands = np.array(rands, dtype=np.int32)
        randsptr = rands.ctypes.data_as(ctypes.c_char_p)
        batchimgs = np.array(batchimgs, dtype=np.uint8)
        batchimgsptr = batchimgs.ctypes.data_as(ctypes.c_char_p)
        self.lib.process(dataptr, cols, rows, len(rands), batchimgsptr, randsptr)
        return img
