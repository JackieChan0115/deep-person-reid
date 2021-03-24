import ctypes
import cv2
import numpy as np
import copy
import torchvision.transforms as T
import os
import random

def testSincurlImage():
	ll = ctypes.cdll.LoadLibrary
	lib = ll("./SincurlImgTool.so")
	img_ori = cv2.imread('/Users/sytm/Documents/Codes/datasets/Market-1501-v15.09.15/bounding_box_train/0002_c1s2_050846_02.jpg')
	img_ori = cv2.resize(img_ori,(128,256)).astype(np.uint8)
	# toTensor = T.ToTensor()
	# img_ori = toTensor(img_ori);
	img_ori = img_ori.transpose((2, 0, 1))
	c,rows, cols = img_ori.shape
	print(img_ori.shape)
	dataptr = img_ori.ctypes.data_as(ctypes.c_char_p)
	lib.process(dataptr, rows, cols)
	img_ori = img_ori.transpose((1,2,0))
	# cv2.imshow("img",img_ori);
	# cv2.waitKey();

	cv2.imwrite("/Users/sytm/Documents/Codes/Cplus/pycpp/climgtool/result.jpg", img_ori)
	print("save finished")


def testFloodfill():
	filepath = '/Users/sytm/Documents/Codes/datasets/Market-1501-v15.09.15/bounding_box_train'
	dir_indexs = [2945, 8451, 1927, 266, 7309, 5518, 9999, 9104, 4369, 2321, 1556, 1300, 5398, 7064, 5913, 6558, 4768, 289,3233, 10659, 9251, 6309, 2214, 3499, 4654, 8239, 10545, 12850, 11698, 11060, 8373, 12598, 3123, 5172,6075, 5309, 9406, 12608, 6081, 5059, 5701, 6470, 10055, 4295, 4936, 11342, 3793, 213, 598, 8793, 5210,2142, 11616, 6628, 4198, 9063, 2280, 9449, 10215, 5613, 496, 2672, 10099, 5748]

	img1_indexs = dir_indexs[0:32]
	dirs = os.listdir(filepath)

	width = 128
	height = 256
	aim_size = (width, height)

	img1s = []
	N = 32
	for i in range(N):
		img1s.append(cv2.resize(cv2.imread(os.path.join(filepath, dirs[img1_indexs[i]])), aim_size).astype(np.uint8))

	img = cv2.resize(cv2.imread(os.path.join(filepath, dirs[dir_indexs[32]])), aim_size).astype(np.uint8)
	cv2.imwrite("/Users/sytm/Documents/Codes/Cplus/pycpp/climgtool/ori_result.jpg", img)
	ll = ctypes.cdll.LoadLibrary
	lib = ll("./FloodfillTool.so")
	rows, cols,c = img.shape
	print(img.shape)
	dataptr = img.ctypes.data_as(ctypes.c_char_p)
	batchimgs = np.asarray(img1s,dtype=np.uint8)
	rands=[]
	for i in range(10):
		rands.append(random.randint(0,31))

	rands = np.asarray(rands, dtype = np.int32)

	batchimgsptr = batchimgs.ctypes.data_as(ctypes.c_char_p)
	randsptr=rands.ctypes.data_as(ctypes.c_char_p)
	lib.process(dataptr, cols,rows,10,batchimgsptr,randsptr)
	cv2.imwrite("/Users/sytm/Documents/Codes/Cplus/pycpp/climgtool/result.jpg", img)
	print("save finished")

testFloodfill();




