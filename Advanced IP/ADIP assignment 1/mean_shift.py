import numpy as np
import cv2
import operator
import sys
import random
import math

def readImage(path_to_image):
	color_image = cv2.imread(path_to_image, 1)
	return color_image

def create5DMatrix(color_image):
	rows = color_image.shape[0]
	cols = color_image.shape[1]
	count = 0
	### Each point as (x,y,b,g,r) ###
	point5D_matrix = np.zeros([rows*cols, 5])
	for i in range(rows):
		for j in range(cols):
			point5D_matrix[count,0] = i
			point5D_matrix[count,1] = j
			point5D_matrix[count,2] = color_image[i,j,0]
			point5D_matrix[count,3] = color_image[i,j,1]
			point5D_matrix[count,4] = color_image[i,j,2]
			count += 1
	return point5D_matrix

def findNeighbours(point5D_matrix, curr_index, distance, marked):
	neighbours = [[point5D_matrix[curr_index,0], point5D_matrix[curr_index,1], point5D_matrix[curr_index,2], point5D_matrix[curr_index,3], point5D_matrix[curr_index,4]]]
	for index, point in enumerate(point5D_matrix):
		b0 = point5D_matrix[curr_index,2]
		g0 = point5D_matrix[curr_index,3]
		r0 = point5D_matrix[curr_index,4]
		bi = point[2]
		gi = point[3]
		ri = point[4]
		dist = math.sqrt((b0-bi)*(b0-bi) + (g0-gi)*(g0-gi) + (r0-ri)*(r0-ri))
		if dist <= distance:
			neighbours.append([point[0], point[1], point[2], point[3], point[4]])
			marked[index] = 1
	return np.asarray(neighbours)

def calculateMean(neighbours, point5D_matrix):
	b_mean = np.mean(neighbours[:,2])
	g_mean = np.mean(neighbours[:,3])
	r_mean = np.mean(neighbours[:,4])

	return [int(b_mean), int(g_mean), int(r_mean)]


def meanShiftFilter(color_image, h, max_iter):

	point5D_matrix = create5DMatrix(color_image)
	print(point5D_matrix.shape[0])

	final_image = color_image.copy()
	rows = color_image.shape[0]
	cols = color_image.shape[1]

	iteration = 1
	while iteration <= max_iter:
		print(iteration)
		marked = [0]*(rows*cols)
		for index, point in enumerate(point5D_matrix):
			if marked[index] is not 1:
				marked[index] = 1
				neighbours = findNeighbours(point5D_matrix, index, h, marked)
				print(index)

				mean = calculateMean(neighbours, point5D_matrix)

				point5D_matrix[index, 2] = mean[0]	
				point5D_matrix[index, 3] = mean[1]
				point5D_matrix[index, 4] = mean[2]

				for neighbour in neighbours:
					final_image[int(neighbour[0]),int(neighbour[1]),0] = point5D_matrix[index,2]
					final_image[int(neighbour[0]),int(neighbour[1]),1] = point5D_matrix[index,3]
					final_image[int(neighbour[0]),int(neighbour[1]),2] = point5D_matrix[index,4]

		iteration += 1

	return final_image

if __name__ == "__main__":
	path_to_image = sys.argv[1]
	h = int(sys.argv[2])
	max_iter = int(sys.argv[3])

	color_image = readImage(path_to_image)
	final_image = meanShiftFilter(color_image, h, max_iter)
	cv2.imwrite('meanShiftFilter.jpg', final_image)

