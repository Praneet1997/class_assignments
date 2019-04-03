import numpy as np
import cv2

def readImage(path_to_image):
	image = cv2.imread(path_to_image)
	return image

def getFeatures(image):
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(image,None)
	return kp, des

def getCorrespondingPoints(des1, des2):	
	bf = cv2.BFMatcher()
	matches = bf.match(des1, des2)
	matches = sorted(matches, key = lambda x:x.distance)
	return matches

def estimateF(X_a, X_b):
	A = np.zeros((X_a.shape[0]**2, X_a.shape[1]))
	z = 0
	for i in range(3):
		for j in range(3):
			A[z,:] = X_b[i,:]*X_a[j,:]
			z += 1
	
	A = A.T
	u,s,vh = np.linalg.svd(A)
	F = np.reshape(vh[-1,:], (3,3))

	u,s,vh = np.linalg.svd(F)
	s[-1] = 0
	S = np.diag(s)
	F = np.matmul(np.matmul(u,S),vh)
	return F


# def RANSAC():
	

if __name__ == '__main__':
	image_names = ['AF1.jpg', 'AF2.jpg'] #two images only
	images = []
	for image_name in image_names:
		image = readImage(image_name)
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		images.append(gray)

	list_of_kp = []
	list_of_des = []

	for image in images:
		kp, des = getFeatures(image)
		list_of_kp.append(kp)
		list_of_des.append(des)

	matches = getCorrespondingPoints(list_of_des[0], list_of_des[1])

	X_a = np.ones((3,len(matches)))
	X_b = np.ones((3,len(matches)))

	for i, match_pt in enumerate(matches):
		x_a = list_of_kp[0][match_pt.queryIdx].pt[0]
		y_a = list_of_kp[0][match_pt.queryIdx].pt[1]
		x_b = list_of_kp[1][match_pt.trainIdx].pt[0]
		y_b = list_of_kp[1][match_pt.trainIdx].pt[1]
		X_a[:,i] = np.array([x_a, y_a, 1])
		X_b[:,i] = np.array([x_b, y_b, 1])
	estimateF(X_a, X_b)

	# img = images[0]
	# img = cv2.drawMatches(images[0], list_of_kp[0], images[1], list_of_kp[1], matches, img, flags=2)
	# cv2.imshow('matched', img)
	# cv2.waitKey(0)