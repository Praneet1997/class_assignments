import numpy as np
import cv2
import operator
import sys

def readImage(path_to_image):
	color_image = cv2.imread(path_to_image, 1)
	return color_image

def colorToGray(color_image):
	rows = color_image.shape[0]
	cols = color_image.shape[1]
	gray = np.zeros((rows, cols, 1), np.float32)

	red = color_image[:,:,2]
	green = color_image[:,:,1]
	blue = color_image[:,:,0]
	#for i in range(rows):
	#	for j in range(cols):
	#		gray[i,j,0] = 0.2126*color_image[i,j,2] + 0.7152*color_image[i,j,1] + 0.0722*color_image[i,j,0]
	gray[:,:,0] = 0.2126*red + 0.7152*green + 0.0722*blue

	cv2.imwrite('gray.jpg', gray)
	return gray

def ifValidPoint(l, m, i, j, rows, cols):
	if (l+i)>=0 and (l+i)<rows and (m+j)>=0 and (m+j)<cols:
		return 1
	else:
		return 0

def harrisResponseCalculator(Ixx, Iyy, Ixy, rows, cols, k, threshold, kernel_size):
	offset = (int)(kernel_size/2)
	harris_corners = []
	R_max = 0
	Rs = np.zeros((rows,cols))
	Rss = np.zeros((rows,cols))
	for x in range(rows):
		for y in range(cols):
			kernelIxx = []
			kernelIyy = []
			kernelIxy = []
			for i in range(-1*offset, offset+1):
				for j in range(-1*offset, offset+1):
					if ifValidPoint(x, y, i, j, rows, cols):
						kernelIxx.append(Ixx[x+i, y+j])
						kernelIyy.append(Iyy[x+i, y+j])
						kernelIxy.append(Ixy[x+i, y+j])
			sumIxx = sum(kernelIxx)
			sumIyy = sum(kernelIyy)
			sumIxy = sum(kernelIxy)

			det = sumIxx*sumIyy - 2*sumIxy
			trace = sumIxx + sumIyy


			R = det - k*trace*trace
			# if R>threshold:
			# 	Rs[x,y] = R
			# 	harris_corners.append([x,y])
			if R > R_max:
				R_max = R
			Rs[x,y] = R
	for r in range(rows):
		for s in range(cols):
			if Rs[r,s] >= threshold*R_max:
				harris_corners.append([r,s])
				Rss[r,s] = Rs[r,s]
				
	return harris_corners, Rss

def nonMaximumSuppression(harris_corners, Rs, rows, cols, nmax_sup_kernel):
	offset = (int)(nmax_sup_kernel/2)
	final_corners = []
	for point_index, point in enumerate(harris_corners):
		x = point[0]
		y = point[1]
		corners_in_kernel = []
		cornerRs_in_kernel = []
		for i in range(-1*offset, offset+1):
			for j in range(-1*offset, offset+1):
				if ifValidPoint(x, y, i, j, rows, cols):
					if [x+i, y+j] in harris_corners:
						cornerRs_in_kernel.append(Rs[x+i, y+j])
						corners_in_kernel.append([x+i, y+j])

		R_max_index, R_max = max(enumerate(cornerRs_in_kernel), key=operator.itemgetter(1))
		final_corners.append(corners_in_kernel[R_max_index])
		
		for corner in corners_in_kernel:
			if corner in harris_corners:
				harris_corners.remove(corner)
	return final_corners
		


def harrisCornerDetector(gray_image, color_image, k, threshold, kernel_size, nmax_sup_kernel):
	### Computing gradient ###
	img = np.resize(gray_image, [gray_image.shape[0], gray_image.shape[1]])
	derivatives = np.gradient(img)

	dx = derivatives[0]
	dy = derivatives[1]

	### Computing M tensor ###
	Ixx = np.multiply(dx,dx)
	Iyy = np.multiply(dy,dy)
	Ixy = np.multiply(dx,dy)

	rows = gray_image.shape[0]
	cols = gray_image.shape[1]
	final_image = color_image.copy()

	harris_corners, Rs = harrisResponseCalculator(Ixx, Iyy, Ixy, rows, cols, k, threshold, kernel_size)
	final_corners = nonMaximumSuppression(harris_corners, Rs, rows, cols, nmax_sup_kernel)
	print(len(harris_corners))
	for point in final_corners:
		final_image[point[0], point[1], 0] = 0
		final_image[point[0], point[1], 1] = 0
		final_image[point[0], point[1], 2] = 255
		#cv2.circle(final_image, (point[0], point[1]), 1, (0,255,0), -1)
	cv2.imwrite('final_image.jpg', final_image)
	return final_image

if __name__ == '__main__':
	path_to_image = sys.argv[1]
	k = float(sys.argv[2])
	threshold = float(sys.argv[3])
	kernel_size = int(sys.argv[4])
	nmax_sup_kernel = int(sys.argv[5])
	color_image = readImage(path_to_image)
	gray_image = colorToGray(color_image)
	final_image = harrisCornerDetector(gray_image, color_image, k, threshold, kernel_size, nmax_sup_kernel)

