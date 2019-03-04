import cv2
import numpy as np
import sys
import math
from scipy import ndimage

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
	gray[:,:,0] = 0.2126*red + 0.7152*green + 0.0722*blue

	cv2.imwrite('gray.jpg', gray)
	return gray

def gaussianFiltering(image, kernel_size):
	rows, cols, ch = image.shape
	new_image = image.copy()
	offset = int(kernel_size/2)
	sigma = 4

	x, y, z = np.meshgrid(np.arange(-1*offset, offset+1), np.arange(-1*offset, offset+1), 0)

	normal = 1 / math.sqrt(2.0 * np.pi * sigma**2)
	kernel = np.exp(-(x**2+y**2) / (2.0*sigma**2)) / normal
	kernel = kernel/np.sum(kernel)

	for i in range(rows - kernel_size + 1):
		for j in range(cols - kernel_size + 1):
			img = image[i:i+kernel_size, j:j+kernel_size]
			window = img * kernel
			new_image[i,j,0] = np.sum(window)
	cv2.imwrite('gauss.jpg', new_image)
	return new_image

def ifValidPoint(l, m, i, j, rows, cols):
	if (l+i)>=0 and (l+i)<rows and (m+j)>=0 and (m+j)<cols:
		return 1
	else:
		return 0

def laplacian(image):
	img = np.resize(image, [image.shape[0], image.shape[1]])
	derivatives = np.gradient(img)

	dx = derivatives[0]
	dy = derivatives[1]

	G = np.sqrt(dx*dx + dy*dy)
	mask = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]])

	Log_image = ndimage.convolve(img, mask, mode='constant')
	Log_image = np.resize(Log_image, [Log_image.shape[0], Log_image.shape[1], 1])

	### Zero crossing ###
	rows, cols, ch = image.shape

	zero_crossing_image = np.zeros((rows,cols,1), np.float32)

	for i in range(rows):
		for j in range(cols):
			if Log_image[i,j,0] < 0:
				for k in range(-1,2):
					for l in range(-1,2):
						if ifValidPoint(k,l,i,j,rows,cols) and Log_image[i+k,j+l,0] > 0:
							zero_crossing_image[i,j,0] = image[i,j,0]
	return Log_image,zero_crossing_image

if __name__ == '__main__':
	path_to_image = sys.argv[1]
	color_image = readImage(path_to_image)
	gray_image = colorToGray(color_image)
	gaussian_filter_image = gaussianFiltering(gray_image, 3)
	final_image, zero_crossing_image = laplacian(gaussian_filter_image)
	cv2.imwrite("LoG image.jpg", final_image)
	cv2.imwrite("zero_crossing_image.jpg", zero_crossing_image)

