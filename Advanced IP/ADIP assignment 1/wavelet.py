import cv2
import numpy as np
import sys
import math

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

def addGaussianNoise(gray_image):
	row,col,ch= gray_image.shape
	mean = 0
	stddev = 5 
	gaussian = np.random.normal(mean,stddev,(row,col,ch))
	gaussian = gaussian.reshape(row,col,ch)
	noisy = gray_image + gaussian
	return noisy

def downSample(image, R):
	rows, cols, ch = image.shape

	### Column-wise ###
	if R == 0:
		img_dnsample = np.zeros((rows, int(cols/2), 1), np.float32)
		for index in range(rows):
			count = 0
			for col in range(0, cols, 2):
				img_dnsample[index, count, 0] = image[index, col, 0]
				count += 1
		return img_dnsample

	elif R == 1:
		img_dnsample = np.zeros((int(rows/2), cols, 1), np.float32)
		for index in range(cols):
			count = 0
			for row in range(0, rows, 2):
				img_dnsample[count, index, 0] = image[row, index, 0]
				count += 1
		return img_dnsample

def upSample(image, R):
	rows, cols, ch = image.shape

	if R == 0:
		img_upsample = np.zeros((rows, cols*2, 1), np.float32)
		for index in range(rows):
			count = 0
			for col in range(0, cols*2, 2):
				img_upsample[index, col, 0] = image[index, count, 0]
				img_upsample[index, col+1, 0] = image[index, count, 0]
				count += 1
		return img_upsample

	elif R == 1:
		img_upsample = np.zeros((rows*2, cols, 1), np.float32)
		for index in range(cols):
			count = 0
			for row in range(0, rows*2, 2):
				img_upsample[row, index, 0] = image[count, index, 0]
				img_upsample[row+1, index, 0] = image[count, index, 0]
				count += 1
		return img_upsample

def convolution(image, R, h):
	rows, cols, ch = image.shape
	### Column - wise ###
	if R == 0:
		imgFilt = np.zeros((cols, max(rows,len(h)), 1))
		for col in range(cols):
			row = np.transpose(image[:,col,0])
			cov = np.convolve(row,h, mode='same')
			imgFilt[col, :, 0] = cov/np.sum(np.absolute(h))
		return np.transpose(imgFilt, (1,0,2))

	elif R == 1:
		imgFilt = np.zeros((rows, max(cols,len(h)), 1))
		for row in range(rows):
			col = image[row,:,0]
			cov = np.convolve(col,h, mode='same')
			imgFilt[row,:,0] = cov/np.sum(np.absolute(h))
		return imgFilt

def waveletTransform(input_image, low_pass_filter, high_pass_filter):
	rows, cols, ch = input_image.shape

	### ROW ###
	imgFilt_low_ROW = convolution(input_image, 1, low_pass_filter)
	lpROW = downSample(imgFilt_low_ROW, 1)

	imgFilt_high_ROW = convolution(input_image, 1, high_pass_filter)
	hpROW = downSample(imgFilt_high_ROW, 1)

	### COL ###
	imgFilt_low_low_COL = convolution(lpROW, 0, low_pass_filter)
	lplpCOL = downSample(imgFilt_low_low_COL, 0)
	#print(lplpCOL.shape)

	imgFilt_high_low_COL = convolution(hpROW, 0, low_pass_filter)
	hplpCOL = downSample(imgFilt_high_low_COL, 0)
	#print(hplpCOL.shape)

	imgFilt_low_high_COL = convolution(lpROW, 0, high_pass_filter)
	lphpCOL = downSample(imgFilt_low_high_COL, 0)
	#print(lphpCOL.shape)

	imgFilt_high_high_COL = convolution(hpROW, 0, high_pass_filter)
	hphpCOL = downSample(imgFilt_high_high_COL, 0)
	#print(hphpCOL.shape)

	return [lplpCOL, hplpCOL, lphpCOL, hphpCOL]

def inverseWaveletTransform(transforms, low_pass_filter, high_pass_filter):
	
	lplpCOL = transforms[0]
	hplpCOL = transforms[1]
	lphpCOL = transforms[2]
	hphpCOL = transforms[3]

	### COL ###

	lplpCOL_UP = upSample(lplpCOL, 0)
	hplpCOL_UP = upSample(hplpCOL, 0)
	lphpCOL_UP = upSample(lphpCOL, 0)
	hphpCOL_UP = upSample(hphpCOL, 0)

	inv_lplp = convolution(lplpCOL_UP, 0, low_pass_filter)
	inv_hplp = convolution(hplpCOL_UP, 0, low_pass_filter)

	inv_lphp = convolution(lphpCOL_UP, 0, high_pass_filter)
	inv_hphp = convolution(hphpCOL_UP, 0, high_pass_filter)


	### ROW ###
	lplpCOL_UP = upSample(inv_lplp, 1)
	hplpCOL_UP = upSample(inv_hplp, 1)
	lphpCOL_UP = upSample(inv_lphp, 1)
	hphpCOL_UP = upSample(inv_hphp, 1)

	inv_lplp = convolution(lplpCOL_UP, 1, low_pass_filter)
	inv_hplp = convolution(hplpCOL_UP, 1, low_pass_filter)

	inv_lphp = convolution(lphpCOL_UP, 1, high_pass_filter)
	inv_hphp = convolution(hphpCOL_UP, 1, high_pass_filter)

	final_image = np.add(np.add(inv_lplp,inv_hplp), np.add(inv_lphp,inv_hphp))
	return final_image

def ifValidPoint(l, m, i, j, rows, cols):
	if (l+i)>=0 and (l+i)<rows and (m+j)>=0 and (m+j)<cols:
		return 1
	else:
		return 0

def medianFilter(image, kernel_size):
	rows, cols, ch = image.shape
	new_image = np.zeros((rows, cols, 1), np.float32)
	offset = int(kernel_size/2)
	for i in range(rows):
		for j in range(cols):
			currPixelNeighbours = []
			for l in range(-1*offset,offset+1):
				for m in range(-1*offset, offset+1):
					if ifValidPoint(l,m,i,j,rows,cols):
						currPixelNeighbours.append(image[i+l, j+m, 0])
			currPixelNeighbours.sort()
			median = currPixelNeighbours[int(len(currPixelNeighbours)/2)]
			new_image[i,j,0] = median
	return new_image

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
	return new_image

if __name__ == "__main__":
	path_to_image = sys.argv[1]
	color_image = readImage(path_to_image)
	gray_image = colorToGray(color_image)
	noisy_image = addGaussianNoise(gray_image)

	cv2.imwrite("noisy_image.jpg", noisy_image)

	analysis_lpf = [-0.125, 0.25, 0.75, 0.25, -0.125]
	analysis_hpf = [-0.5, 1, 0.5]

	synthesis_lpf = [0.5, 1, 0.5]
	synthesis_hpf = [-0.125, 0.25, 0.75, 0.25, 0.125]
	transforms = waveletTransform(noisy_image, analysis_lpf, analysis_hpf)

	wavelet_analysis_image = inverseWaveletTransform(transforms, synthesis_lpf, synthesis_hpf)
	median_filter_image = medianFilter(noisy_image, 3)

	gaussian_filter_image = gaussianFiltering(noisy_image, 5)

	cv2.imwrite('median_filter_image.jpg', median_filter_image)
	cv2.imwrite("wavelet_analysis_image.jpg", wavelet_analysis_image)
	cv2.imwrite('gaussian_filter_image.jpg', gaussian_filter_image)
