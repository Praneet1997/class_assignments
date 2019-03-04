import cv2
import sys
import math
import numpy as np
from scipy import linalg

xi, yi = -1, -1
distMode = False
get_points = np.zeros((4,2))
count = 0

def readImage(path_to_image):
	color_image = cv2.imread(path_to_image, 1)
	return color_image

def pixelPosition(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print('Position of the pixel is:')
		print('x = ', x)
		print('y = ', y)

def distaceBtPixels(event, x, y, flags, param):
	global distMode
	if event == cv2.EVENT_LBUTTONDOWN:
			global xi, yi 
			xi, yi = x, y
			distMode = True

	if event == cv2.EVENT_RBUTTONDOWN and distMode == True:
		distance = math.sqrt((xi-x)*(xi-x) + (yi-y)*(yi-y))
		print(distance)
		distMode = False

def getPoints(event, x, y, flags, param):
	global get_points, count
	if event == cv2.EVENT_LBUTTONDOWN and count < 4:
		get_points[count,:] = [x,y]
		count += 1



def vanishingPoint(point_set1, point_set2):
	points = [point_set1, point_set2]
	lines = []
	for point in points:
		A0, A1, A2 = point[0,0], point[0,1], 1
		B0, B1, B2 = point[1,0], point[1,1], 1
		C0 = A1*B2 - A2*B1
		C1 = A2*B0 - A0*B2
		C2 = A0*B1 - A1*B0
		lines.append([C2, C0, C1])

	Dx = np.linalg.det(np.array([[-lines[0][0], lines[0][2]],[-lines[1][0], lines[1][2]]]))
	Dy = np.linalg.det(np.array([[lines[0][1], -lines[0][0]],[lines[1][1], -lines[1][0]]]))
	D = np.linalg.det(np.array([[lines[0][1], lines[0][2]],[lines[1][1], lines[1][2]]]))

	x = Dx/D
	y = Dy/D


	return (x,y,1)

def createAMatrix(img_points, world_points):
	A = np.zeros((2*img_points.shape[0], 12))

	for i in range(img_points.shape[0]):
		A[2*i, :] = [0, 0, 0, 0, -img_points[i][2]*world_points[i][0], -img_points[i][2]*world_points[i][1], -img_points[i][2]*world_points[i][2], -img_points[i][2]*world_points[i][3], img_points[i][1]*world_points[i][0], img_points[i][1]*world_points[i][1], img_points[i][1]*world_points[i][2], img_points[i][1]*world_points[i][3]]
		A[2*i+1, :] = [img_points[i][2]*world_points[i][0], img_points[i][2]*world_points[i][1], img_points[i][2]*world_points[i][2], img_points[i][2]*world_points[i][3], 0, 0, 0, 0, -img_points[i][0]*world_points[i][0], -img_points[i][0]*world_points[i][1], -img_points[i][0]*world_points[i][2], -img_points[i][0]*world_points[i][3]]
	return A

def projection1():
	img_points = np.zeros((9,3))

	img_points[0,0] = 159
	img_points[0,1] = 404
	img_points[0,2] = 1

	img_points[1,0] = 168 
	img_points[1,1] = 244
	img_points[1,2] = 1

	img_points[2,0] = 389 
	img_points[2,1] = 357
	img_points[2,2] = 1

	img_points[3,0] = 393 
	img_points[3,1] = 241
	img_points[3,2] = 1

	img_points[4,0] = 532 
	img_points[4,1] = 328
	img_points[4,2] = 1

	img_points[5,0] = 531 
	img_points[5,1] = 240
	img_points[5,2] = 1

	img_points[6,0] = 517 
	img_points[6,1] = 394
	img_points[6,2] = 1

	img_points[7,0] = 625 
	img_points[7,1] = 359
	img_points[7,2] = 1

	img_points[8,0] = 620 
	img_points[8,1] = 313
	img_points[8,2] = 1

	world_points = np.zeros((9,4))

	world_points[0,0] = 0 
	world_points[0,1] = 0
	world_points[0,2] = 0
	world_points[0,3] = 1

	world_points[1,0] = 0 
	world_points[1,1] = 2
	world_points[1,2] = 0
	world_points[1,3] = 1

	world_points[2,0] = 4
	world_points[2,1] = 0
	world_points[2,2] = 0
	world_points[2,3] = 1

	world_points[3,0] = 4 
	world_points[3,1] = 2
	world_points[3,2] = 0
	world_points[3,3] = 1

	world_points[4,0] = 8 
	world_points[4,1] = 0
	world_points[4,2] = 0
	world_points[4,3] = 1

	world_points[5,0] = 8 
	world_points[5,1] = 2
	world_points[5,2] = 0
	world_points[5,3] = 1

	world_points[6,0] = 4 
	world_points[6,1] = 0
	world_points[6,2] = 1.5
	world_points[6,3] = 1

	world_points[7,0] = 8 
	world_points[7,1] = 0
	world_points[7,2] = 1.5
	world_points[7,3] = 1

	world_points[8,0] = 12 
	world_points[8,1] = 0
	world_points[8,2] = 0
	world_points[8,3] = 1

	A = createAMatrix(img_points, world_points)
	u, s, v = np.linalg.svd(A)
	v = np.reshape(v[v.shape[0]-1, :], (3,4))

	file = open("output.txt","w") 
	for i in range(world_points.shape[0]):
		file.write(str(world_points[i,0]) + " " + str(world_points[i,1]) + " " + str(world_points[i,2])+ "->")
		file.write(str(img_points[i,0]) + " "  + str(img_points[i,1])+ "\n")
	
	file.close()

	return v

def projection2():
	real_pts = np.ones((9,4))	
	image_pts = np.ones((9,3))

	real_pts[0,0] = real_pts[0,1] = real_pts[0,2] = 0

	real_pts[1,0] = 4
	real_pts[1,1] = real_pts[1,2] = 0

	real_pts[2,0] = 8
	real_pts[2,1] = real_pts[2,2] = 0

	real_pts[3,0] = 12
	real_pts[3,1] = real_pts[3,2] = 0

	real_pts[4,0] = 16
	real_pts[4,1] = real_pts[4,2] = 0

	real_pts[5,2] = -2
	real_pts[5,0] = real_pts[5,1] = 0

	real_pts[6,2] = -6
	real_pts[6,0] = real_pts[6,1] = 0

	real_pts[7,2] = -7.5
	real_pts[7,0] = real_pts[7,1] = 0

	real_pts[8,2] = -12
	real_pts[8,0] = real_pts[8,1] = 0

	image_pts[0,0] = 106
	image_pts[0,1] = 123
	image_pts[1,0] = 173
	image_pts[1,1] = 121
	image_pts[2,0] = 255
	image_pts[2,1] = 119
	image_pts[3,0] = 334
	image_pts[3,1] = 121
	image_pts[4,0] = 419
	image_pts[4,1] = 121
	image_pts[5,0] = 107
	image_pts[5,1] = 155
	image_pts[6,0] = 107
	image_pts[6,1] = 199
	image_pts[7,0] = 109
	image_pts[7,1] = 220
	image_pts[8,0] = 112
	image_pts[8,1] = 279

	A = createAMatrix(image_pts, real_pts)
	u, s, v = np.linalg.svd(A)
	v = np.reshape(v[v.shape[0]-1, :], (3,4))

	file = open("output.txt","w") 
	for i in range(real_pts.shape[0]):
		file.write(str(real_pts[i,0]) + " " + str(real_pts[i,1]) + " " + str(real_pts[i,2]) + " -> ")
		file.write(str(image_pts[i,0]) + " "  + str(image_pts[i,1])+ "\n")
	
	file.close()

	return v 

def cameraMatrix(P):
	M = P[:,:3]
	K, R = linalg.rq(M, mode='full')
	C = -1*np.matmul(np.linalg.inv(M), P[:,3])
	extrinsic = np.zeros((3,4))
	extrinsic[0:3, 0:3] = R
	extrinsic[:,3] = -1*np.matmul(R,C)
	return K, extrinsic

if __name__ == "__main__":
	path_to_image = sys.argv[1]
	mode = sys.argv[2]

	color_image = readImage(path_to_image)
	cv2.namedWindow('image')
	if mode == '0':
		cv2.setMouseCallback('image', pixelPosition)
	elif mode == '1':
	 	cv2.setMouseCallback('image', distaceBtPixels)
	elif mode == '2':
		cv2.setMouseCallback('image', getPoints)

	while(1):
		global count, get_points
		cv2.imshow('image', color_image)

		if cv2.waitKey(20) & 0xFF == 27:
			break

		if count == 4 and mode=='2':
			point_set1 = get_points[:2,:]
			point_set2 = get_points[2:4,:]
			vanishing_point = vanishingPoint(point_set1, point_set2)
			print(vanishing_point)
			break

		if mode == '3' and path_to_image == 'Helipad.jpg':
			v = projection1()
			K, extrinsic = cameraMatrix(v)
			print('Projection matrix:')
			print(v)
			print('Intrinsic:')
			print(K)
			print('extrinsic:')
			print(extrinsic)
			break

		if mode == '3' and path_to_image == 'Palace.jpg':
			v = projection2()
			K, extrinsic = cameraMatrix(v)
			print('Projection matrix:')
			print(v)
			print('Intrinsic:')
			print(K)
			print('extrinsic:')
			print(extrinsic)
			break

