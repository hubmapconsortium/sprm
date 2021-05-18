import numpy as np
import os
from os.path import join
from scipy.sparse import csr_matrix
from skimage.io import imread
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import variation
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.morphology import closing
from skimage.morphology import diameter_closing

"""

Companion to SPRM.py
Package functions that evaluate a single segmentation method
Author: Haoran Chen
05/17/2021


"""

def fraction(img_bi, mask_bi):
	foreground_all = np.sum(img_bi)
	background_all = img_bi.shape[0] * img_bi.shape[1] - foreground_all
	mask_all = np.sum(mask_bi)
	background = len(np.where(mask_bi - img_bi == 1)[0])
	foreground = np.sum(mask_bi * img_bi)
	return foreground / foreground_all, background / background_all, foreground / mask_all


def foreground_separation(img):
	from skimage.filters import threshold_mean
	from skimage import measure
	threshold = threshold_mean(img)
	img_sep = img > threshold
	img_sep = img_sep.astype(int)
	img_sep = closing(img_sep, disk(3))
	img_sep = -img_sep + 1
	img_sep = diameter_closing(img_sep, 5)
	labels = measure.label(img_sep, background=0, connectivity=2)
	label_num = np.unique(labels)
	for i in label_num:
		label_loc = np.where(labels == i)
		if i == 0:
			labels[label_loc] = 0
		elif len(label_loc[0]) > 5000:
			labels[label_loc] = -1
		else:
			labels[label_loc] = 0
	labels = labels + 1
	return labels


def uniformity_CV(loc, channels):
	CV = []
	n = len(channels)
	for i in range(n):
		channel = channels[i]
		channel = channel / np.mean(channel)
		background_intensity = channel[tuple(loc.T)]
		CV.append(np.std(background_intensity))
	return np.average(CV)


def uniformity_fraction(loc, channels):
	n = len(channels)
	for i in range(n):
		channel = channels[i]
		channel = channel / np.mean(channel)
		background_intensity = channel[tuple(loc.T)]
		if i == 0:
			feature_matrix = background_intensity
		else:
			feature_matrix = np.vstack((feature_matrix, background_intensity))
	model = PCA(n_components=1).fit(feature_matrix)
	fraction = model.explained_variance_ratio_[0]
	return fraction


def foreground_uniformity(img_bi, mask, channels):
	foreground_loc = np.argwhere((img_bi - mask) == 1)
	CV = uniformity_CV(foreground_loc, channels)
	fraction = uniformity_fraction(foreground_loc, channels)
	return CV, fraction


def background_uniformity(img_bi, mask, channels):
	background_loc = np.argwhere((img_bi + mask) == 0)
	CV = uniformity_CV(background_loc, channels)
	fraction = uniformity_fraction(background_loc, channels)
	return CV, fraction


def cell_uniformity_CV(feature_matrix):
	CV = []
	for i in range(feature_matrix.shape[1]):
		CV.append(variation(feature_matrix[:, i]))
	CV_avg = np.nanmean(CV)
	return CV_avg


def cell_uniformity_fraction(feature_matrix):
	model = PCA(n_components=1).fit(feature_matrix.T)
	fraction = model.explained_variance_ratio_[0]
	return fraction

def weighted_by_cluster(vector, labels):
	for i in range(len(vector)):
		# print(vector[i])
		# print(len(np.where(labels == i)[0]))
		vector[i] = vector[i] * len(np.where(labels == i)[0])
	weighted_average = np.sum(vector) / len(labels)
	return weighted_average


def cell_uniformity(mask, channels):
	cell_index = np.unique(mask)
	n = len(channels)
	cell_coord = get_indices_sparse(mask)
	for i in range(n):
		# channel = imread(os.path.join(data_dir, 'R001_X001_Y001.ome-' + str(i) + '.tif'))
		channel = channels[i]
		channel = channel / np.mean(channel)
		cell_intensity = []
		for j in range(1, len(cell_index)):
			cell_intensity.append(np.sum(channel[tuple(cell_coord[j])]))
		if i == 0:
			feature_matrix = np.array(cell_intensity)
		else:
			feature_matrix = np.vstack((feature_matrix, cell_intensity))
	feature_matrix = feature_matrix.T
	CV = []
	fraction = []
	silhouette = []
	for c in range(1, 11):
		model = KMeans(n_clusters=c).fit(feature_matrix)
		labels = model.labels_
		CV_current = []
		fraction_current = []
		if c == 1:
			silhouette.append(1)
		else:
			try:
				silhouette.append(silhouette_score(feature_matrix, labels))
			except:
				silhouette.append(0)
		
		for i in range(c):
			cluster_feature_matrix = feature_matrix[np.where(labels == i)[0], :]
			CV_current.append(cell_uniformity_CV(cluster_feature_matrix))
			fraction_current.append(cell_uniformity_fraction(cluster_feature_matrix))
		CV.append(weighted_by_cluster(CV_current, labels))
		fraction.append(weighted_by_cluster(fraction_current, labels))
	return CV, fraction, silhouette


def get_matched_cells(arr1, arr2):
	a = set((tuple(i) for i in arr1))
	b = set((tuple(i) for i in arr2))
	if len(list(b - a)) == 0:
		return np.array(list(a)), np.array(list(b))
	else:
		return False, False


def compute_M(data):
	cols = np.arange(data.size)
	return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
	M = compute_M(data)
	return [np.unravel_index(row.data, data.shape) for row in M]


def show_plt(mask):
	plt.imshow(mask)
	plt.show()
	plt.clf()


def list_remove(c_list, indexes):
	for index in sorted(indexes, reverse=True):
		del c_list[index]
	return c_list


def filter_cells(coords, mask):
	# completely mismatches
	no_cells = []
	for i in range(len(coords)):
		if np.sum(mask[coords[i]]) == 0:
			no_cells.append(i)
	new_coords = list_remove(coords.copy(), no_cells)
	return new_coords



def get_indexed_mask(mask, boundary):
	boundary = boundary * 1
	boundary_loc = np.where(boundary == 1)
	boundary[boundary_loc] = mask[boundary_loc]
	return boundary


def get_boundary(mask):
	mask_boundary = find_boundaries(mask)
	mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
	return mask_boundary_indexed


def get_mask(cell_list, mask_shape):
	mask = np.zeros((mask_shape))
	for cell_num in range(len(cell_list)):
		mask[tuple(cell_list[cell_num].T)] = cell_num
	return mask


def get_matched_mask(whole_cell_mask, nuclear_mask):
	

	cell_coords = get_indices_sparse(whole_cell_mask)[1:]
	nucleus_coords = get_indices_sparse(nuclear_mask)[1:]
	
	cell_coords = list(map(lambda x: np.array(x).T, cell_coords))
	nucleus_coords = list(map(lambda x: np.array(x).T, nucleus_coords))
	
	cell_matched_index_list = []
	nucleus_matched_index_list = []
	cell_matched_list = []
	nucleus_matched_list = []
	
	for i in range(len(cell_coords)):
		if (i not in cell_matched_index_list) and len(cell_coords[i]) != 0:
			current_cell_coords = cell_coords[i]
			nuclear_search_num = np.unique(list(map(lambda x: nuclear_mask[tuple(x)], current_cell_coords))) - 1
			for j in nuclear_search_num:
				if j != -1 and j != 65535 and (j not in nucleus_matched_index_list):
					whole_cell, nucleus = get_matched_cells(cell_coords[i], nucleus_coords[j])
					if type(whole_cell) != bool:
						cell_matched_list.append(whole_cell)
						nucleus_matched_list.append(nucleus)
						cell_matched_index_list.append(i)
						nucleus_matched_index_list.append(j)
	
	cell_matched_mask = get_mask(cell_matched_list, whole_cell_mask.shape)
	nuclear_matched_mask = get_mask(nucleus_matched_list, whole_cell_mask.shape)
	return cell_matched_mask, nuclear_matched_mask


def single_method_eval(img, mask, result_dir):
	print('Calculating single-method metrics...')
	
	# get best z slice for future use
	bestz = mask.bestz
	
	# separate image foreground background
	img_binary = foreground_separation(np.sum(np.squeeze(img.data[0, 0, :, bestz, :, :], axis=0), axis=0))

	# set mask channel names
	channel_names = ['cells', 'nucleus', 'cell_membrane', 'nuclear_membrane']
	
	for channel in range(mask.data.shape[2]):
		
		# set output directory for current mask channel
		channel_dir = join(result_dir, 'single_method_metric', channel_names[channel])
		if not os.path.exists(channel_dir):
			os.makedirs(channel_dir)
			
		# calculate mismatch metric if whole-cell mask
		if channel == 0:
			whole_cell_mask = np.squeeze(mask.data[0, 0, 0, bestz, :, :], axis=0)
			nuclear_mask = np.squeeze(mask.data[0, 0, 1, bestz, :, :], axis=0)
			cell_matched_mask, nuclear_matched_mask = get_matched_mask(whole_cell_mask, nuclear_mask)
			whole_cell_mask_binary = np.sign(whole_cell_mask)
			nuclear_mask_binary = np.sign(nuclear_mask)
			cell_matched_mask_binary = np.sign(cell_matched_mask)
			nuclear_matched_mask_binary = np.sign(nuclear_matched_mask)
			total_area = np.sum(np.sign(whole_cell_mask_binary + nuclear_mask_binary))
			mismatched_area = np.sum(np.sign((nuclear_mask_binary - nuclear_matched_mask_binary) + (whole_cell_mask_binary - cell_matched_mask_binary)))
			mismatched_fraction = mismatched_area / total_area
			np.savetxt(join(channel_dir, 'compartments_mismatched_fraction.txt'), [mismatched_fraction])
		
		# get current mask
		current_mask = np.squeeze(mask.data[0, 0, channel, bestz, :, :], axis=0)
		# get current binary mask
		mask_binary = np.sign(current_mask)
		
		# calculate number of cell
		cell_num = len(np.unique(current_mask)) - 1
		
		# get fraction of image occupied by the mask
		mask_fraction = 1 - (len(np.where(current_mask == 0)[0]) / (current_mask.shape[0] * current_mask.shape[1]))
		
		# get coverage metrics
		foreground_fraction, background_fraction, mask_foreground_fraction = fraction(img_binary, mask_binary)
		np.savetxt(join(channel_dir, 'cell_basic.txt'), [cell_num, mask_fraction, foreground_fraction, background_fraction, mask_foreground_fraction])
		
		img_channels = np.squeeze(img.data[0, 0, :, bestz, :, :], axis=0)
		# get background and foreground uniformity
		foreground_CV, foreground_PCA = foreground_uniformity(img_binary, mask_binary, img_channels)
		background_CV, background_PCA = background_uniformity(img_binary, mask_binary, img_channels)
		np.savetxt(join(channel_dir, 'foreground_uniformity.txt'), [foreground_CV, foreground_PCA])
		np.savetxt(join(channel_dir, 'background_uniformity.txt'), [background_CV, background_PCA])
		
		# get cell uniformity
		cell_CV, cell_fraction, cell_silhouette = cell_uniformity(current_mask, img_channels)
		np.savetxt(join(channel_dir, 'cell_uniformity.txt'), [cell_CV, cell_fraction, cell_silhouette])


