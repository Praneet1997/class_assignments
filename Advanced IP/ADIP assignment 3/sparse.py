import numpy as np
import random
import cv2
import sys
from skimage.util.shape import view_as_windows
import scipy
import os
import math
from scipy.optimize import linear_sum_assignment

def readImage(path_to_image):
	image = cv2.imread(path_to_image)
	return image

def getPatches(image, Y, m=8):
	rows, cols, ch = image.shape
	patch = np.zeros((m,m,3), np.uint8)
	used_centers = []
	#flatten_patches = []

	# i = 0
	# while i < 1000:
	# 	image_padded = np.pad(image, ((3,3), (3,3), (0,0)), 'symmetric')
	# 	n_x = random.randint(m/2,rows-1-m/2)
	# 	n_y = random.randint(m/2,cols-1-m/2)
	# 	if (n_x,n_y) not in used_centers:
	# 		patch[:,:,:] = image_padded[n_x-m/2+1:n_x+m/2+1, n_y-m/2+1:n_y+m/2+1, :]
	# 		flatten_patch = patch.flatten()
	# 		flatten_patches.append(flatten_patch)
	# 		used_centers.append((n_x,n_y))
	# 		i = i+1

	new_patches = view_as_windows(image,(m,m,3), m)
	flatten_patches = []
	for i in range(new_patches.shape[0]):
		for j in range(new_patches.shape[1]):
			used_centers.append([i*8,j*8])
			Y.append(np.transpose(np.squeeze(new_patches[i,j], axis=0), (2,0,1)).flatten())

	return Y, used_centers

def sparseCoding(Yj, D, L):
	Aj = np.zeros((D.shape[1]), np.float32)
	indices = []
	error = np.sum(Yj-D.dot(Aj))
	#print(Yj.shape)
	while error > 0.1 and len(indices) <= L:
		atom_contrib = D.T.dot(Yj - D.dot(Aj))
		index = np.argmax(atom_contrib)
		#print(atom_contrib.shape)
		indices.append(index)
		Aj[index] += atom_contrib[index]

		D_influencing = D[:,indices]
		Aj_updated = np.linalg.pinv(D_influencing).dot(Yj)
		Aj[indices] = Aj_updated
		error = np.sum(Yj-D.dot(Aj))

	return Aj


def dictionaryUpdate(Y, D, A):
	print('Dictionary Update')
	for j in range(D.shape[1]):
		indices = np.where(A[j,:])
		if(len(indices[0])>1):
			D[:,j] = 0
			E = Y - D.dot(A)
			E_influenced = E[:,indices[0]]
			u, s, vh = scipy.sparse.linalg.svds(E_influenced, k=1)
			D[:,j] = u[:,0]
			A_influenced = vh[0]*(s[0])
			A[j,indices] = A_influenced
	return D, A

def reconstructImage(D,A,m,used_centers,images):
	print('-------------- Reconstructing image --------------')
	Y = D.dot(A)
	Y = np.asarray(Y, dtype=np.uint8)
	countPatches = 0
	reconstructed_images = []
	reconstructed_img_sizes = []
	

	for i, centers in enumerate(used_centers):
		reconstructed_image = np.zeros((images[i].shape[0], images[i].shape[1], 3), np.uint8)
		for center in centers:
			n_x = center[0]
			n_y = center[1]
			reconstructed_image[n_x:n_x+m, n_y:n_y+m, 0] = Y[0:m*m,countPatches].reshape((m,m))
			reconstructed_image[n_x:n_x+m, n_y:n_y+m, 1] = Y[m*m:m*m*2,countPatches].reshape((m,m))
			reconstructed_image[n_x:n_x+m, n_y:n_y+m, 2] = Y[m*m*2:,countPatches].reshape((m,m))
			countPatches += 1 
		reconstructed_images.append(reconstructed_image)
		
	return reconstructed_images

def getCompressionRatio(reconstructed_img_sizes, original_img_sizes):
	for i in range(len(reconstructed_img_sizes)):
		print('Compression Ratio for image ' + str(i+1) + ' is ', float(original_img_sizes[i])/float(reconstructed_img_sizes[i]))

def getPSNR(images, reconstructed_images):
	for index in range(len(images)):
		image = images[index]
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		reconstructed_image = reconstructed_images[index]
		reconstructed_image_gray = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2GRAY)
		MSE = np.mean((image_gray-reconstructed_image_gray)**2)
		if MSE == 0.0:
			return 100
		else:
			PSNR = 20*math.log10(255/math.sqrt(MSE))
			print('PSNR of image ' + str(index+1) + ' is ', PSNR)

def bipartiteMatching(dictionaries, A_matrices, frame_centers, dataset_images, datasets_image_sizes):
	dict1 = dictionaries[0]
	dict2 = dictionaries[1]
	cost_matrix = np.zeros((dict1.shape[1], dict2.shape[1]))

	for i in range(dict1.shape[1]):
		for j in range(dict2.shape[1]):
			cost_matrix[i,j] = np.linalg.norm(dict1[:,i]-dict2[:,j])

	row_ind, col_ind = linear_sum_assignment(cost_matrix)

	new_dict1 = dict1
	new_dict2 = dict2

	new_dict1[:, row_ind] = dict2[:, col_ind]
	new_dict2[:, col_ind] = dict1[:, row_ind]

	dictionaries = [new_dict1, new_dict2]

	for i in range(len(dictionaries)):
		reconstructed_img_sizes = []
		reconstructed_images = reconstructImage(dictionaries[i], A_matrices[i], 8, frame_centers[i], dataset_images[i])
		for j, reconstructed_image in enumerate(reconstructed_images):
			cv2.imwrite('bipartiteMatching_dataset_' + str(i) + '/reconstructed_transformed_'+ str(j) + '_' + str(i) + '.jpg', reconstructed_image)
			b = os.path.getsize('bipartiteMatching_dataset_' + str(i) + '/reconstructed_transformed_'+ str(j) + '_' + str(i) + '.jpg')
			reconstructed_img_sizes.append(b)
		print(len(datasets_image_sizes[i]))
		getCompressionRatio(reconstructed_img_sizes, datasets_image_sizes[i])
		getPSNR(dataset_images[i], reconstructed_images)




if __name__=='__main__':

	datasets = ['dataset_1.txt', 'dataset_2.txt']
	dataset_images = []
	dictionaries = []
	A_matrices = []
	frame_centers = []
	datasets_image_sizes = []

	for dataset_index, file in enumerate(datasets):
		Y = []
		images = []
		used_centers = []
		original_img_sizes = []

		f = open(file, 'r')
		path_to_image = f.readline().rstrip()

		for image_name in f:
			print(path_to_image + image_name)
			image = readImage(path_to_image + image_name.rstrip())
			#image = cv2.resize(image, (image.shape[1]/2, image.shape[0]/2))
			b = os.path.getsize(path_to_image + image_name.rstrip())
			#print(b)
			original_img_sizes.append(b)
			images.append(image)
		dataset_images.append(images)
		
		for image in images:
			Y, centers = getPatches(image, Y, 8)
			used_centers.append(centers)

		frame_centers.append(used_centers)

		Y = np.asarray(Y, dtype=np.float32)
		Y = Y.T

		#initialize D
		k = 256
		random_index = np.random.rand(256,1)
		random_index = random_index*Y.shape[1]
		random_index = np.asarray(random_index, dtype='int')

		D = np.asarray([Y[:,i] for i in random_index], dtype=np.float32)
		D = D.reshape((D.shape[0], D.shape[1]))
		D = D.T

		A = np.zeros((D.shape[1], Y.shape[1]), np.float32)

		for iteration in range(15):
			print('--------------Iter ' + str(iteration) + ' --------------')
			print('sparseCoding')
			for j in range(A.shape[1]):
				Aj = sparseCoding(Y[:,j], D, 5)
				A[:,j] = Aj
			D,A = dictionaryUpdate(Y,D,A)

		reconstructed_images = reconstructImage(D,A,8,used_centers,images)
		reconstructed_img_sizes = []

		for i, reconstructed_image in enumerate(reconstructed_images):
			cv2.imwrite('reconstructed_dataset_' + str(dataset_index) + '/reconstructed_'+ str(i) + '_' + str(dataset_index) + '.jpg', reconstructed_image)
			b = os.path.getsize('reconstructed_dataset_' + str(dataset_index) + '/reconstructed_'+ str(i) + '_' + str(dataset_index) + '.jpg')
			reconstructed_img_sizes.append(b)

		getCompressionRatio(reconstructed_img_sizes, original_img_sizes)
		getPSNR(images, reconstructed_images)

		datasets_image_sizes.append(original_img_sizes)
		dictionaries.append(D)
		A_matrices.append(A)

	bipartiteMatching(dictionaries, A_matrices, frame_centers, dataset_images, datasets_image_sizes)

	
	



	


	
	
