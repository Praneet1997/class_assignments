import numpy as np
import random
import cv2
import sys
from skimage.util.shape import view_as_windows

def readImage(path_to_image):
	image = cv2.imread(path_to_image)
	return image

def getPatches(image, m=8):
	rows, cols, ch = image.shape
	patch = np.zeros((m,m,3), np.uint8)
	used_centers = []
	flatten_patches = []
	Y = []

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
			flatten_patches.append(np.transpose(np.squeeze(new_patches[i,j], axis=0), (2,0,1)).flatten())

	Y = np.asarray(flatten_patches, dtype=np.float32)
	Y = Y.T
	return Y, used_centers

def sparseCoding(Y, D, L): #using OMP algorithm
	print('Sparse Coding')
	A = np.zeros((D.shape[1], Y.shape[1]), np.float32)

	for j in range(A.shape[1]):
		indices = []
		error = np.sum(Y-D.dot(A)) 
		while  error > 0.1 and len(indices) <= L:
			atom_contrib = D.T.dot(Y[:,j] - D.dot(A[:,j]))
			index = np.argmax(atom_contrib)
			indices.append(index)
			A[index,j] = atom_contrib[index]
			D_influencing = D[:,indices]
			D_inv = np.linalg.pinv(D_influencing)
			Aj_updated = D_inv.dot(Y[:,j])
			A[indices,j] = Aj_updated
			error = np.sum(Y-D.dot(A))
	return A

def dictionaryUpdate(Y, D, A):
	print('Dictionary Update')

	for j in range(D.shape[1]):
		indices = np.where(A.T[:,j])
		if(len(indices[0])>0):
			D[:,j] = 0
			E = Y - D.dot(A)
			E_influenced = E.T[indices[0], :]
			E_influenced = E_influenced.T

			A_influenced = A.T[indices,j]
		
			Dj = D[:,j]
			Dj = Dj.reshape(Dj.shape[0],1)
			res = Dj.dot(A_influenced) - E_influenced

			u, s, vh = np.linalg.svd(res)

			D[:,j] = u[:,0]
			A_influenced = vh[:,0].dot(s[0])

			A.T[indices,j] = A_influenced.T

	return D, A

def reconstructImage(D,A,m,used_centers,image):
	Y = D.dot(A)
	reconstructed_image = np.zeros((image.shape[0], image.shape[1], 3), np.float32)

	for index, centers in enumerate(used_centers):
		n_x = centers[0]
		n_y = centers[1]
		reconstructed_image[n_x:n_x+m, n_y:n_y+m, 0] = Y[0:m*m,index].reshape((m,m))
		reconstructed_image[n_x:n_x+m, n_y:n_y+m, 1] = Y[m*m:m*m*2,index].reshape((m,m))
		reconstructed_image[n_x:n_x+m, n_y:n_y+m, 2] = Y[m*m*2:,index].reshape((m,m)) 

	cv2.imshow('reconstructed_image_b', reconstructed_image/255)
	cv2.waitKey(0)

if __name__=='__main__':
	path_to_image = sys.argv[1]
	m = int(sys.argv[2])

	image = readImage(path_to_image)
	image = cv2.resize(image, (image.shape[1]/2, image.shape[0]/2))
	Y, used_centers = getPatches(image, m)

	#initialize D
	k = 256
	random_index = np.random.rand(256,1)
	random_index = random_index*Y.shape[1]
	random_index = np.asarray(random_index, dtype='int')

	D = np.asarray([Y[:,i] for i in random_index], dtype=np.float32)
	D = D.reshape((D.shape[0], D.shape[1]))
	D = D.T

	for iteration in range(5):
		A = sparseCoding(Y, D, 5)
		D,A = dictionaryUpdate(Y,D,A)
		print('error', np.sum((Y-D.dot(A))/Y))

	reconstructImage(D,A,m,used_centers,image)
	#reconstructImage(D,m,used_centers,image,Y)
	
	



	


	
	
