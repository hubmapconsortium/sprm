import importlib.resources
import numpy as np
import os
from os.path import join
from PIL import Image
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
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
"""

Companion to SPRM.py
Package functions that evaluate a single segmentation method
Author: Haoran Chen and Ted Zhang
Version: 1.0
06/09/2021


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
	threshold = threshold_mean(img.astype(np.int64))
	img_sep = img > threshold
	img_sep = img_sep.astype(int)
	img_sep = closing(img_sep, disk(5))
	img_sep = -img_sep + 1
	labels = measure.label(img_sep, background=0, connectivity=2)
	label_num = np.unique(labels)
	label_coords = get_indices_sparse(labels)
	for i in label_num:
		label_loc = label_coords[i]
		if i == 0:
			labels[label_loc] = 1
		elif len(label_loc[0]) > 20000:
			labels[label_loc] = 0
		else:
			labels[label_loc] = 1
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
	cell_coord = get_indices_sparse(mask)[1:]
	for i in range(n):
		# channel = imread(os.path.join(data_dir, 'R001_X001_Y001.ome-' + str(i) + '.tif'))
		channel = channels[i]
		channel = channel / np.mean(channel)
		cell_intensity = []
		for j in range(len(cell_index)-1):
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

def flatten_dict(input_dict):
	local_list = []
	for key, value in input_dict.items():
		if type(value) == dict:
			local_list.extend(flatten_dict(value))
		else:
			local_list.append(value)
	return local_list

def single_method_eval(img, mask, output_dir: Path):
	print('Calculating single-method metrics for', img.path)

	# get best z slice for future use
	bestz = mask.bestz
	
	# get compartment masks
	matched_mask = np.squeeze(mask.data[0, 0, :, bestz, :, :], axis=0)
	cell_matched_mask = matched_mask[0]
	matched_cell_num = len(np.unique(cell_matched_mask))
	nuclear_matched_mask = matched_mask[1]
	cell_membrane_mask = matched_mask[2]
	nuclear_membrane_mask = matched_mask[3]
	no_mem_cell_matched_mask = cell_matched_mask - cell_membrane_mask
	no_mem_nuclear_matched_mask = nuclear_matched_mask - nuclear_membrane_mask
	cytoplasm_mask = (no_mem_cell_matched_mask - nuclear_matched_mask)
	cytoplasm_mask[np.where(cytoplasm_mask >= matched_cell_num)] = 0
	cytoplasm_mask[np.where(cytoplasm_mask < 0)] = 0


	metric_mask = np.expand_dims(cell_matched_mask, 0)
	metric_mask = np.vstack((metric_mask, np.expand_dims(cell_membrane_mask, 0)))
	metric_mask = np.vstack((metric_mask, np.expand_dims(cytoplasm_mask, 0)))
	metric_mask = np.vstack((metric_mask, np.expand_dims(nuclear_membrane_mask, 0)))
	metric_mask = np.vstack((metric_mask, np.expand_dims(no_mem_nuclear_matched_mask, 0)))
	# separate image foreground background
	img_binary = sum(foreground_separation(np.squeeze(img.data[0, 0, c, bestz, :, :], axis=0)) for c in range(img.data.shape[2]))
	img_binary = np.sign(img_binary)
	np.savetxt(output_dir / f'{img.name}_img_binary.txt.gz', img_binary)
	fg_bg_image = Image.fromarray(img_binary.astype(np.uint8) * 255, mode='L').convert('1')
	fg_bg_image.save(output_dir / f'{img.name}_img_binary.png')

	# set mask channel names
	channel_names = ['matched_cells', 'cell_membrane', 'cytoplasm', 'nuclear_membrane', 'nucleus_no_mem']
	
	metrics = {}
	for channel in range(metric_mask.shape[0]):
		# print('--Calculating ' + channel_names[channel] + ' channel')
		current_mask = metric_mask[channel]
		mask_binary = np.sign(current_mask)
		metrics[channel_names[channel]] = {}
		if channel == 0:
			# read fraction of match metric from metadata
			im_metadata = img.img.metadata.to_xml()
			mask_metadata = mask.img.metadata.to_xml()

			tag = mask_metadata.find('FractionOfMatchedCellsAndNuclei')
			if tag is None:
				matched_fraction = 1.0
			else:
				matched_fraction = float(tag)
			# matched_fraction = float(root[1][0][0][0][1].text)
	
			# get pixel size in squared micron
			start_ind = im_metadata.find('PhysicalSizeX=') + len('PhysicalSizeX=') + 1
			pixel_size = float(im_metadata[start_ind:start_ind+5]) ** 2

			pixel_num = mask_binary.shape[0] * mask_binary.shape[1]
			micron_num = pixel_size * pixel_num
			
			# calculate number of cell per 100 squared micron
			cell_num = len(np.unique(current_mask)) - 1
			cell_num_normalized = cell_num / micron_num * 100
			
			# get fraction of image occupied by the mask
			mask_fraction = 1 - (len(np.where(current_mask == 0)[0]) / (current_mask.shape[0] * current_mask.shape[1]))
			
			# get coverage metrics
			foreground_fraction, background_fraction, _ = fraction(img_binary, mask_binary)
			
			metrics[channel_names[channel]]['NumberOfCellPer100SquareMicrons'] = cell_num_normalized
			metrics[channel_names[channel]]['FractionOfImageOccupiedByCells'] = mask_fraction
			metrics[channel_names[channel]]['FractionOfForegroundOccupiedByCells'] = foreground_fraction
			metrics[channel_names[channel]]['1-FractionOfBackgroundOccupiedByCells'] = 1 - background_fraction
			metrics[channel_names[channel]]['FractionOfMatchedCellsAndNuclei'] = matched_fraction
			
		else:
			_, _, mask_foreground_fraction = fraction(img_binary, mask_binary)
			img_channels = np.squeeze(img.data[0, 0, :, bestz, :, :], axis=0)
			# get background and foreground uniformity
			foreground_CV, foreground_PCA = foreground_uniformity(img_binary, mask_binary, img_channels)
			background_CV, background_PCA = background_uniformity(img_binary, mask_binary, img_channels)
			
			# get cell uniformity
			cell_CV, cell_fraction, cell_silhouette = cell_uniformity(current_mask, img_channels)
			avg_cell_CV = np.average(cell_CV)
			avg_cell_fraction = np.average(cell_fraction)
			avg_cell_silhouette = np.average(cell_silhouette)

			metrics[channel_names[channel]]['FractionOfMaskInForeground'] = mask_foreground_fraction
			metrics[channel_names[channel]]['1/AvgCVForegroundOutsideCells'] = 1 / foreground_CV
			metrics[channel_names[channel]]['FractionOfFirstPCForegroundOutsideCells'] = foreground_PCA
			metrics[channel_names[channel]]['1/AvgCVBackgroundOutsideCells'] = 1 / background_CV
			metrics[channel_names[channel]]['FractionOfFirstPCBackgroundOutsideCells'] = background_PCA
			metrics[channel_names[channel]]['1/AvgOfWeightedAvgCVCellIntensitiesOver2~10NumberOfCluster'] = 1 / avg_cell_CV
			metrics[channel_names[channel]]['AvgOfWeightedAvgFractionOfFirstPCCellIntensitiesOver2~10NumberOfCluster'] = avg_cell_fraction
			metrics[channel_names[channel]]['AvgOfWeightedAvgSilhouetteOver2~10NumberOfCluster'] = avg_cell_silhouette

	metrics_flat = np.expand_dims(flatten_dict(metrics), 0)
	# print(metrics_flat)

	with importlib.resources.open_binary('sprm', 'pca.pickle') as f:
		ss, pca = pickle.load(f)

	metrics_flat_scaled = ss.transform(metrics_flat)
	pca_score = pca.transform(metrics_flat_scaled)[0, 0]
	metrics['QualityScore'] = pca_score

	return metrics
