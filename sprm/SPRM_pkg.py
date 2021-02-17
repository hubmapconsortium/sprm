from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
import re
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import time
from sklearn.decomposition import PCA
import pandas as pd
from itertools import product, chain, combinations
import math
from typing import Dict, List, Any, Sequence, Union
from pathlib import Path
from .outlinePCA import shape_cluster
from skimage.filters import threshold_otsu
from .ims_sparse_allchan import findpixelfractions
from .ims_sparse_allchan import reallocateIMS as reallo
import multiprocessing
# from joblib import Parallel, delayed
from skimage.feature.texture import greycomatrix, greycoprops
import numba as nb
from numba.typed import Dict as nbDict
# from numba.typed import List as nbList
from sklearn.metrics import silhouette_score

"""

Companion to SPRM.py
Package functions that are integral to running main script
Author: Ted Zhang & Robert F. Murphy
01/21/2020 - 02/17/2020
Version: 0.80


"""

'''
Global vars
'''
INTEGER_PATTERN = re.compile(r'(\d+)')
FILENAMES_TO_IGNORE = frozenset({'.DS_Store'})
num_cores = multiprocessing.cpu_count()


class IMGstruct:

    def __init__(self, path: Path, options):
        self.img = self.read_img(path, options)
        self.data = self.read_data(options)
        self.path = path
        self.name = path.name
        self.channel_labels = self.read_channel_names()

    def set_data(self, data):
        self.data = data

    def set_img(self, img):
        self.img = img

    def get_data(self):
        return self.data

    def get_meta(self):
        return self.img.metadata

    def quit(self):
        return self.img.close()

    @staticmethod
    def read_img(path: Path, options: Dict) -> AICSImage:
        img = AICSImage(path)
        if not img.metadata:
            print('Metadata not found in input image')
            # might be a case-by-basis
            # img = AICSImage(path), known_dims="CYX")

        return img

    def read_data(self, options):
        data = self.img.data[:, :, :, :, :, :]
        dims = data.shape
        s, t, c, z, y, x = dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
        if t > 1:
            data = data.reshape((s, 1, t * c, z, y, x))

        return data

    def read_channel_names(self):
        img = self.img
        return img.get_channel_names(scene=0)

    def get_name(self):
        return self.name

    def get_channel_labels(self):
        return self.channel_labels

    def set_channel_labels(self, channel_list):
        self.channel_labels = channel_list


class MaskStruct(IMGstruct):

    def __init__(self, path: Path, options):
        super().__init__(path, options)
        self.bestz = self.get_bestz()
        self.interior_cells = []
        self.edge_cells = []
        self.cell_index = []
        self.bad_cells = []

    def get_labels(self, label):
        return self.channel_labels.index(label)

    def set_bestz(self, z):
        self.bestz = z

    def get_bestz(self):
        return self.bestz

    def read_data(self, options):
        bestz = []
        data = self.img.data
        dims = data.shape
        # s,t,c,z,y,x = dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]
        check = data[:, :, :, 0, :, :]
        check_sum = np.sum(check)
        if check_sum == 0:  # assumes the best z is not the first slice
            print('Duplicating best z to all z dimensions...')
            for i in range(0, data.shape[3]):
                x = data[:, :, :, i, :, :]
                y = np.sum(x)
                #                print(x)
                #                print(y)
                if y > 0:
                    bestz.append(i)
                    break
                else:
                    continue

            if options.get("debug"): print('Best z dimension found: ', bestz)
            # data now contains just the submatrix that has nonzero values
            # add back the z dimension
            data = x[:, :, :, np.newaxis, :, :]
            # print(data.shape)
            # and replicate it
            data = np.repeat(data, dims[3], axis=3)
            # print(data.shape)
            # set bestz
        else:
            bestz.append(0)

        # set bestz
        self.set_bestz(bestz)

        return data

    def add_edge_cell(self, ncell):
        self.edge_cells.append(ncell)

    def set_edge_cells(self, listofcells):
        self.edge_cells = listofcells

    def get_edge_cells(self):
        return self.edge_cells

    def set_interior_cells(self, listofincells):
        self.interior_cells = listofincells

    def get_interior_cells(self):
        return self.interior_cells

    def set_cell_index(self, cellindex):
        self.cell_index = cellindex

    def get_cell_index(self):
        return self.cell_index

    def set_bad_cells(self, bcells):
        self.bad_cells = bcells

    def get_bad_cells(self):
        return self.bad_cells

def calculations(coord, im: IMGstruct, t: int, i: int) -> (np.ndarray, np.ndarray, np.ndarray):
    '''
        Returns covariance matrix, mean vector, and total vector
    '''

    if i == 0: print('Performing statistical analyses on ROIs...')

    y, x = coord[1], coord[0]
    y = y.astype(int)
    x = x.astype(int)

    temp = im.get_data()

    channel_all_mask = temp[0, t, :, 0, x, y]
    ROI = np.transpose(channel_all_mask)

    cov_m = np.cov(ROI)
    mu_v = np.reshape(np.mean(ROI, axis=1), (ROI.shape[0], 1))
    total = np.reshape(np.sum(ROI, axis=1), (ROI.shape[0], 1))

    # filter for NaNs
    cov_m[np.isnan(cov_m)] = 0
    mu_v[np.isnan(mu_v)] = 0
    total[np.isnan(total)] = 0

    # if not cov_m.shape:
    # cov_m = np.array([cov_m])

    return cov_m, mu_v, total


def cell_cluster_format(cell_matrix: np.ndarray, segnum: int, options: Dict) -> np.array:
    '''
    Receives out_matrix and extracts information to output a vector: 
    len # of cells with corresponding cluster number
    '''
    # extracting all cells through all timepoints --> 4D matrix 
    # cell_matrix = outmatrix[:,0,:,:,:]
    # dims = get_dims(cell_matrix)
    # t, cl, cha, chb  = dims[0], dims[1], dims[2], dims[3]
    if cell_matrix.shape[0] > 1:
        print('cell_matrix has more than one time point')
        print('Have not implemented support yet..')
        exit()
    cell_matrix = cell_matrix[0, :, :, :]
    # optional: pull out one seg method
    if segnum >= 0:
        cell_matrix = cell_matrix[segnum, :, :, :]
        cell_matrix = cell_matrix.reshape((cell_matrix.shape[0], cell_matrix.shape[1] * cell_matrix.shape[2]))
        if options.get("debug"):
            print('segmentation channel: ' + str(segnum + 1))
            print('3D cell matrix')
            print(cell_matrix.shape)
    else:
        # temp_matrix = np.zeros([cell_matrix.shape[1], cell_matrix.shape[2], cell_matrix.shape[3], cell_matrix.shape[0]])
        # for i in range(0, cell_matrix.shape[0]):
        #     temp_matrix[:, :, :, i] = cell_matrix[i, :, :, :]
        # cell_matrix2 = temp_matrix
        # cell_matrix2 = cell_matrix2.reshape(
        #     (cell_matrix2.shape[0], cell_matrix2.shape[1] * cell_matrix2.shape[2] * cell_matrix2.shape[3]))

        cell_matrix = cell_matrix.reshape(
            (cell_matrix.shape[1], cell_matrix.shape[2] * cell_matrix.shape[3] * cell_matrix.shape[0]))

    return cell_matrix


def cell_cluster(cell_matrix: np.ndarray, options: Dict) -> (np.ndarray, np.ndarray):
    # kmeans clustering
    print('Clustering cells...')
    # check of clusters vs. n_sample wanted
    cluster_method, num_cellclusters = options.get("num_cellclusters")
    if num_cellclusters > cell_matrix.shape[0]:
        print('reducing cell clusters to ', cell_matrix.shape[0])
        num_cellclusters = cell_matrix.shape[0]

    # skipping clustering because all 0s of texture
    if options.get('texture_flag'):
        cellbycluster = KMeans(n_clusters=1, random_state=0).fit(cell_matrix)
        options.pop('texture_flag', None)
    else:

        if cluster_method == 'silhouette':
            cluster_list = []
            cluster_score = []
            for i in range(2, num_cellclusters + 1):
                cellbycluster = KMeans(n_clusters=i, random_state=0)
                preds = cellbycluster.fit_predict(cell_matrix)
                cluster_list.append(cellbycluster)

                score = silhouette_score(cell_matrix, preds)
                cluster_score.append(score)

            max_value = max(cluster_score)
            idx = cluster_score.index(max_value)

            cellbycluster = cluster_list[idx]
            cellbycluster = cellbycluster.fit(cell_matrix)

        else:
            cellbycluster = KMeans(n_clusters=num_cellclusters, random_state=0).fit(cell_matrix)

    # returns a vector of len cells and the vals are the cluster numbers
    cellbycluster_labels = cellbycluster.labels_
    clustercenters = cellbycluster.cluster_centers_
    if options.get("debug"):
        print(clustercenters.shape)
        print(cellbycluster_labels.shape)

    return cellbycluster_labels, clustercenters


# def map: cell index to cluster index. Mask indexed img changes that to cluster number (mask,cellbycluster)
def cell_map(mask: MaskStruct, cc_v: np.ndarray, seg_n: int, options: Dict) -> np.ndarray:
    '''
    Maps the cells to indexed img
    '''
    print('Mapping...')
    mask_img = mask.get_data()
    mask_img = mask_img[0, 0, seg_n, :, :, :]
    # temp = mask_img.copy()
    temp = np.zeros(mask_img.shape)
    # cluster_img = np.zeros(mask_img.shape)
    # start_time = time.monotonic()
    cc_v += 1
    clusters = np.unique(cc_v)

    # for i in range(0, len(cc_v)):
    #     coord = np.where(mask_img == i + 1)
    #     cluster_img[coord[0], coord[1], coord[2]] = cc_v[i] + 1
    # # cluster_imgt = [mask_img == i+1]
    # # cluster_imgt = cluster_img * (cc_v[i]+1)
    # elapsed_time2 = time.monotonic() - start_time

    stime = time.monotonic() if options.get("debug") else None
    for i in range(0, len(clusters)):
        cell_num = np.where(cc_v == clusters[i])[0] + 1
        bit_mask = np.isin(mask_img, cell_num)
        temp[bit_mask] = clusters[i]

    if options.get("debug"): print('Elapsed time for cell mapping <vectorized>: ', time.monotonic() - stime)
    # print('Elapsed time for cell mapping <loop>: ', time.monotonic() - start_time)
    return temp
    # return cluster_img


# @nb.njit(parallel=True)
def append_coord(masked_imgs_coord, rlabel_mask, indices):
    for i in range(0, len(rlabel_mask)):
        masked_imgs_coord[rlabel_mask[i]][0].append(indices[0][i])
        masked_imgs_coord[rlabel_mask[i]][1].append(indices[1][i])

    return


@nb.njit(parallel=True)
def cell_num_index_map(flat_mask: np.ndarray, cell_num_dict: Dict):
    for i in nb.prange(0, len(flat_mask)):
        flat_mask[i] = cell_num_dict.get(flat_mask[i])


def unravel_indices(mask_channels, maxvalue, channel_coords):
    for j in range(0, len(mask_channels)):
        masked_imgs_coord = [[[], []] for i in range(maxvalue)]
        labeled_mask = mask_channels[j]
        rlabel_mask = labeled_mask[0, :, :].reshape(-1)
        indices = np.arange(len(rlabel_mask))
        indices = np.unravel_index(indices, (labeled_mask.shape[1], labeled_mask.shape[2]))

        append_coord(masked_imgs_coord, rlabel_mask, indices)
        masked_imgs_coord = list(map(np.asarray, masked_imgs_coord))

        channel_coords.append(masked_imgs_coord)


def npwhere(mask_channels, maxvalue, channel_coords):
    for j in range(len(mask_channels)):
        cmask = mask_channels[j][0]
        d = []
        for k in range(0, maxvalue):
            coords = np.where(cmask == k)
            d.append(coords)
        channel_coords.append(d)


@nb.njit()
def nb_populate_dict(cell_num, cell_num_idx):
    d = nbDict.empty(nb.types.int64, nb.types.int64)

    for i in range(0, len(cell_num)):
        d[cell_num[i]] = cell_num_idx[i]

    return d


def get_coordinates(mask, options):
    mask_channels = []
    channel_coords = []
    # channel_coords_np = []
    s, t, c, z, y, x = mask.get_data().shape
    mask_data = mask.get_data().copy()

    # find cell index - if not sequential
    cell_num = np.unique(mask_data)
    maxvalue = len(cell_num)
    mask.set_cell_index(cell_num)

    if maxvalue != np.max(mask_data):
        cell_num_idx = np.arange(0, len(cell_num))
        # cell_num_dict = dict(zip(cell_num, cell_num_idx))
        cell_num_dict = nb_populate_dict(cell_num, cell_num_idx)
        fmask_data = mask_data.reshape(-1)

        # for i in range(0, len(fmask_data)):
        #     fmask_data[i] = cell_num_dict.get(fmask_data[i])
        cell_num_index_map(fmask_data, cell_num_dict)

        fmask_data = fmask_data.reshape((s, t, c, z, y, x))
        mask.set_data(fmask_data)
        mask_data = mask.get_data()

    #post-process for edge case cell coordinates - only 1 point
    freq = np.unique(mask_data[0, 0, 0, 0, :, :], return_counts=True)
    idx = np.where(freq[1] == 1)[0].tolist()
    mask.set_bad_cells(idx)

    mask_4D = mask_data[0, 0, :, :, :, :]

    for i in range(0, mask_4D.shape[0]):
        mask_channels.append(mask_4D[i, :, :, :])

    unravel_indices(mask_channels, maxvalue, channel_coords)  # new
    # npwhere(mask_channels, maxvalue, channel_coords_np) #old

    #remove idx from coords
    # if len(idx) != 0:
    #     for i in range(len(channel_coords)):
    #         channel_coords[i] = [i for j, i in enumerate(channel_coords[i]) if j not in idx]

    return channel_coords

# def remove_idx(l, idx):
#     del l[idx]

# @nb.njit(parallel=True)
# def test(cell_count, mask_ch, maxcell):
#     for i in nb.prange(0, maxcell):
#         coords = np.where(mask_ch == i)
#         cell_count.append(coords)
#
#
# def get_coordinates_nb(mask):
#     mask_channels = []
#     channel_coords = []
#     mask_4D = mask.get_data()[0, 0, :, :, :, :]
#     maxvalue = np.max(mask_4D) + 1
#
#     for i in range(0, mask_4D.shape[0]):
#         mask_channels.append(mask_4D[i, :, :, :])
#
#     for j in range(len(mask_channels)):
#         cell_count = nbList(lsttype=ListType(UniTuple(int64[:], 2)))
#         cell_count = test(cell_count, mask_channels[j][0], maxvalue)
#
#         channel_coords.append(cell_count)
#
#     return channel_coords
#

# not used anymore
# def mask_img(mask: MaskStruct, j: int) -> (np.ndarray, np.ndarray):
#     '''
#     returns: a 3D matrix that represents the jth segmented image
#     '''
#     print('Getting indexed mask from ' + mask.get_channel_labels()[j] + ' channel')
#     sMask = mask.get_data()
#
#     # print(sMask.shape)
#     sMask = sMask[0, 0, j, :, :, :]
#     # unique = np.unique(sMask)
#     #
#     # edgeCells = mask.get_edge_cells()
#     #
#     # interiorCells = [i for i in unique if i not in edgeCells]
#
#     interiorCells = np.asarray(mask.get_interior_cells())
#
#     return sMask, interiorCells


# not used anymore
def get_masked_imgs(labeled_mask: np.ndarray, maskIDs: np.ndarray) -> List[np.ndarray]:
    '''
        Returns the masked image as a set of coordinates
    '''
    print('Getting coordinates that correspond with ROIs from indexed mask...')
    # add in check for edge cell detection. Could get outofbounds issue

    # masked_imgs_coord = np.zeros((len(maskIDs),2), dtype= np.uint16)
    # masked_imgs_coord = defaultdict(list)
    # labeled_mask = np.int64(labeled_mask)

    maxvalue = np.max(labeled_mask) + 1
    masked_imgs_coord = [[[], []] for i in range(maxvalue)]
    # need to implement a more efficient algo
    # for i in range(0, len(maskIDs)):
    #     coor = np.where(labeled_mask == maskIDs[i])
    #     #masked_imgs_coord[i] = coor

    rlabel_mask = labeled_mask[0, :, :].reshape(-1)
    indices = np.arange(len(rlabel_mask))
    indices = np.unravel_index(indices, (labeled_mask.shape[2], labeled_mask.shape[1]))

    # s = time.monotonic()
    for i in range(0, len(rlabel_mask)):
        masked_imgs_coord[rlabel_mask[i]][0].append(indices[0][i])
        masked_imgs_coord[rlabel_mask[i]][1].append(indices[1][i])

    masked_imgs_coord = list(map(np.asarray, masked_imgs_coord))
    # print(time.monotonic() - s)

    # test = [[] for i in range(maxvalue)]
    #
    # # s1 = time.monotonic()
    # for i in range(0, len(rlabel_mask)):
    #     test[rlabel_mask[i]].append(i)
    #
    # new_coords = []
    # for j in range(0, len(test)):
    #     if test[j]:
    #         new_coords.append([np.take(indices[0], test[j]), np.take(indices[1], test[j])])
    # # print(time.monotonic() - s1)

    ############################################
    # need to implement this check
    # assert(len(maskIDs) == len(new_coords))

    return masked_imgs_coord

    # for i in range(0, labeled_mask.shape[0]):
    #     for j in range(0, labeled_mask.shape[1]):
    #         masked_imgs_coord[labeled_mask[0, i, j]].append((i, j))

    # masked_imgs_coord = find_mask_indices(labeled_mask, maskIDs, masked_imgs_coord)
    # return masked_imgs_coord


def SRM(img_files, mask_files, options):
    '''
        Matlab wrapper for python
    '''
    eng = matlab.engine.start_matlab("-desktop")
    eng.cd('./SRMcode')
    answer = eng.main_HPA(img_files, mask_files, options, nargout=1)
    # print(answer)


def try_parse_int(value: str) -> Union[int, str]:
    if value.isdigit():
        return int(value)
    return value


def alphanum_sort_key(path: Path) -> Sequence[Union[int, str]]:
    """
    By: Matt Ruffalo
    Produces a sort key for file names, alternating strings and integers.
    Always [string, (integer, string)+] in quasi-regex notation.
    >>> alphanum_sort_key(Path('s1 1 t.tiff'))
    ['s', 1, ' ', 1, ' t.tiff']
    >>> alphanum_sort_key(Path('0_4_reg001'))
    ['', 0, '_', 4, '_reg', 1, '']
    """
    return [try_parse_int(c) for c in INTEGER_PATTERN.split(path.name)]


def get_paths(img_dir: Path) -> Sequence[Path]:
    if img_dir.is_dir():
        img_files = [c for c in img_dir.iterdir() if c.name not in FILENAMES_TO_IGNORE]
    else:
        # assume it's a pattern, like Path('some/dir/*.tiff')
        # don't need to filter filenames, because the user passed a
        # glob pattern of exactly what is wanted
        img_files = list(img_dir.parent.glob(img_dir.name))

    return sorted(img_files, key=alphanum_sort_key)


def get_df_format(sub_matrix, s: str, img: IMGstruct, options: Dict) -> (List[str], Any):
    # can use dict switch to speed up access if the formats get > 5
    names = img.get_channel_labels()
    if len(sub_matrix.shape) < 3:
        if 'shape' in s:
            header = [i for i in range(1, sub_matrix.shape[1] + 1)]
        else:
            header = names
    elif 'covar' in s:
        concat = np.reshape(sub_matrix, (sub_matrix.shape[0], sub_matrix.shape[1] * sub_matrix.shape[2]))
        sub_matrix = concat
        comb = []
        for pair in product(names, names):
            comb.append(':'.join(pair))
        header = comb
        options["channel_label_combo"] = header
    else:
        pass
    return header, sub_matrix


def write_2_file(sub_matrix, s: str, img: IMGstruct, output_dir: Path, options: Dict):
    header, sub_matrix = get_df_format(sub_matrix, s, img, options)
    write_2_csv(header, sub_matrix, s, output_dir, options)


def write_2_csv(header: List, sub_matrix, s: str, output_dir: Path, options: Dict):
    global row_index

    if row_index:
        df = pd.DataFrame(sub_matrix, index=options.get('row_index_names'))
    else:
        df = pd.DataFrame(sub_matrix, index=list(range(1, sub_matrix.shape[0] + 1)))
    if options.get("debug"): print(df)
    f = output_dir / (s + '.csv')

    # column header for indices
    df.index.name = 'ID'
    df.to_csv(f, header=header)

    # Sean Donahue - 11/12/20
    key_parts = s.replace('-', '_').split('_')
    key_parts.reverse()
    hdf_key = '/'.join(key_parts)
    with pd.HDFStore(output_dir / Path('out.hdf5')) as store:
        store.put(hdf_key, df)


def write_cell_polygs(polyg_list: List[np.ndarray], filename: str, output_dir: Path, options: Dict):
    coord_pairs = []
    for i in range(0, len(polyg_list)):
        tlist = str([[round(i, 4), round(j, 4)] for i, j in zip(polyg_list[i][:, 0], polyg_list[i][:, 1])])
        coord_pairs.append(tlist)

    df = pd.DataFrame({0: coord_pairs}, index=list(range(1, len(coord_pairs) + 1)))
    if options.get("debug"): print(df)
    f = output_dir / (filename + '-cell_polygons_spatial.csv')
    df.index.name = 'ID'
    df.to_csv(f, header=['Shape'])


def build_matrix(im: IMGstruct, mask: MaskStruct, masked_imgs_coord: List[np.ndarray], j: int,
                 omatrix: np.ndarray) -> np.ndarray:
    if j == 0:
        return np.zeros(
            (im.get_data().shape[1], mask.get_data().shape[2], len(masked_imgs_coord), im.get_data().shape[2],
             im.get_data().shape[2]))
    else:
        return omatrix


def build_vector(im: IMGstruct, mask: MaskStruct, masked_imgs_coord: List[np.ndarray], j: int,
                 omatrix: np.ndarray) -> np.ndarray:
    if j == 0:
        return np.zeros(
            (im.get_data().shape[1], mask.get_data().shape[2], len(masked_imgs_coord), im.get_data().shape[2], 1))
    else:
        return omatrix


def clusterchannels(im: IMGstruct, options: Dict) -> np.ndarray:
    '''
        cluster all channels using PCA
    '''
    print('Dimensionality Reduction of image channels...')
    if options.get("debug"): print('Image dimensions before reduction: ', im.get_data().shape)
    pca_channels = PCA(n_components=options.get("num_channelPCA_components"), svd_solver='full')
    channvals = im.get_data()[0, 0, :, :, :, :]
    keepshape = channvals.shape
    channvals = channvals.reshape(channvals.shape[0], channvals.shape[1] * channvals.shape[2] * channvals.shape[3])
    channvals = channvals.transpose()
    if options.get("debug"): print(channvals.shape)
    reducedim = pca_channels.fit_transform(channvals)
    if options.get("debug"):
        print('PCA Channels: \n', pca_channels)
        print('Explained variance ratio: ', pca_channels.explained_variance_ratio_)
        print('Singular values: ', pca_channels.singular_values_)
        print('Image dimensions before transposing & reshaping: ', reducedim.shape)
    reducedim = reducedim.transpose()
    reducedim = reducedim.reshape(reducedim.shape[0], keepshape[1], keepshape[2], keepshape[3])
    if options.get("debug"): print('Image dimensions after transposing and reshaping: ', reducedim.shape)
    return reducedim


def plotprincomp(reducedim: np.ndarray, bestz: int, filename: str, output_dir: Path, options: Dict) -> np.ndarray:
    print('Plotting PCA image...')
    reducedim = reducedim[:, bestz, :, :]
    k = reducedim.shape[0]
    if k > 2:
        reducedim = reducedim[0:3, :, :]
    else:
        rzeros = np.zeros(reducedim.shape)
        reducedim[k:3, :, :] = rzeros[k:3, :, :]

    plotim = reducedim.transpose(1, 2, 0)

    if options.get("debug"):
        print('Before Transpose:', reducedim.shape)
        print('After Transpose: ', plotim.shape)

    for i in range(0, 3):
        cmin = plotim[:, :, i].min()
        cmax = plotim[:, :, i].max()
        if options.get("debug"): print('Min and Max before normalization: ', cmin, cmax)
        plotim[:, :, i] = ((plotim[:, :, i] - cmin) / (cmax - cmin)) * 255.
        cmin = plotim[:, :, i].min()
        cmax = plotim[:, :, i].max()
        if options.get("debug"): print('Min and Max after normalization: ', cmin, cmax)
    plotim = plotim.round()
    plotim = plotim.astype(int)
    plt.clf()
    plt.imshow(plotim)
    plt.axis('off')
    # plt.show(block=False)
    f = output_dir / (filename)
    plt.savefig(f)
    plt.close()

    return plotim


def SNR(im: IMGstruct, filename: str, output_dir: Path, options: Dict):
    '''
    signal to noise ratio of each channel of the image w/ z score and otsu thresholding
    '''

    global row_index

    print('Calculating Signal to Noise Ratio in image...')
    channvals = im.get_data()[0, 0, :, :, :, :]
    channelist = []

    channvals_z = channvals.reshape(channvals.shape[0], channvals.shape[1] * channvals.shape[2] * channvals.shape[3])
    channvals_z = channvals_z.transpose()

    m = channvals_z.mean(0)
    sd = channvals_z.std(axis=0, ddof=0)
    snr_channels = np.where(sd == 0, 0, m / sd)
    # snr_channels = snr_channels[np.newaxis, :]
    # snr_channels = snr_channels.tolist()

    # define otsu threshold
    for i in range(0, channvals.shape[0]):
        img_2D = channvals[i, :, :, :]
        img_2D = img_2D.reshape((img_2D.shape[0] * img_2D.shape[1], img_2D.shape[2]))

        try:
            thresh = threshold_otsu(img_2D)
            above_th = img_2D >= thresh
            below_th = img_2D < thresh
            if thresh == 0:
                fbr = 1
            else:
                mean_a = np.mean(img_2D[above_th])
                mean_b = np.mean(img_2D[below_th])
                # take ratio above thresh / below thresh
                fbr = mean_a / mean_b
        except ValueError:
            fbr = 1

        channelist.append(fbr)

    channelist = np.asarray(channelist)

    snrs = np.stack([snr_channels, channelist])

    # check for nans
    snrs[np.isnan(snrs)] = 0

    # add index identifier
    row_index = 1
    options['row_index_names'] = ['Z-Score', 'Otsu']

    # write out 2 rows
    write_2_csv(im.get_channel_labels(), snrs, filename + '-SNR', output_dir, options)

    # turn index boolean off
    row_index = 0

    # return snr_channels, channelist


def voxel_cluster(im: IMGstruct, options: Dict) -> np.ndarray:
    '''
    cluster multichannel image into superpixels
    '''
    print('Clustering voxels into superpixels...')
    if im.get_data().shape[0] > 1:
        print('image has more than one time point')
        print('Have not implemented support yet..')
        exit()
    channvals = im.get_data()[0, 0, :, :, :, :]
    keepshape = channvals.shape
    channvals = channvals.reshape(channvals.shape[0], channvals.shape[1] * channvals.shape[2] * channvals.shape[3])
    channvals = channvals.transpose()
    # for some reason, voxel values are occasionally NaNs (especially edge rows)
    channvals[np.where(np.isnan(channvals))] = 0
    if options.get("debug"): print('Multichannel dimensions: ', channvals.shape)
    # get random sampling of pixels in 2d array
    np.random.seed(0)
    sampling = float(options.get('precluster_sampling'))
    samples = math.ceil(sampling * channvals.shape[0])
    # lower bound threshold on pixels
    if samples < options.get("precluster_threshold"):
        samples = options.get("precluster_threshold")
    idx = np.random.choice(channvals.shape[0], samples)
    channvals_random = channvals[idx]
    # kmeans clustering of random sampling
    print('Clustering random sample of voxels...')
    stime = time.monotonic() if options.get("debug") else None

    num_voxelclusters = options.get('num_voxelclusters')
    # if options.get("cluster_evaluation_method") == 'silhouette':
    #     cluster_list = []
    #     cluster_score = []
    #
    #     for i in range(2, num_voxelclusters + 1):
    #         voxelbycluster = KMeans(n_clusters=num_voxelclusters, random_state = 0)
    #         preds = voxelbycluster.fit_predict(channvals_random)
    #         cluster_list.append(voxelbycluster)
    #
    #         score = silhouette_score(channvals_random, preds)
    #         cluster_score.append(score)
    #
    #     max_value = max(cluster_score)
    #     idx = cluster_score.index(max_value)
    #
    #     voxelbycluster = cluster_list[idx]
    #     voxelbycluster = voxelbycluster.fit(channvals_random)
    # else:

    voxelbycluster = KMeans(n_clusters=options.get("num_voxelclusters"), random_state=0).fit(channvals_random)

    if options.get("debug"): print('random sample voxel cluster runtime: ' + str(time.monotonic() - stime))
    cluster_centers = voxelbycluster.cluster_centers_
    # fast kmeans clustering with inital centers
    print('Clustering voxels with initialized centers...')
    stime = time.monotonic() if options.get("debug") else None
    voxelbycluster = KMeans(n_clusters=options.get("num_voxelclusters"), init=cluster_centers, random_state=0,
                            max_iter=100, verbose=0, n_init=1).fit(channvals)
    # voxelbycluster = KMeans(n_clusters=options.get("num_voxelclusters"), random_state=0).fit(channvals)
    if options.get("debug"): print('Voxel cluster runtime: ', time.monotonic() - stime)
    # returns a vector of len number of voxels and the vals are the cluster numbers
    voxelbycluster_labels = voxelbycluster.labels_
    voxelbycluster_labels = voxelbycluster_labels.reshape(keepshape[1], keepshape[2], keepshape[3])

    if options.get("debug"):
        print('Cluster Label dimensions: ', voxelbycluster_labels.shape)
        print('Number of unique labels:')
        print(len(np.unique(voxelbycluster_labels)))

    return voxelbycluster_labels


def findmarkers(clustercenters: np.ndarray, options: Dict) -> List:
    '''
        find a set of markers that have the largest variance without correlations among them
    '''
    clustercenters = np.transpose(clustercenters)
    markerlist = []
    markergoal = options.get('num_markers')
    if clustercenters.shape[1] < 2:
        markerlist = list(np.argsort(-clustercenters.ravel())[:markergoal])
        return markerlist

    if clustercenters.shape[0] < markergoal:
        markergoal = clustercenters.shape[0]
        print("Reducing marker goal to ", markergoal)

    # warning of true_divide / 0 when std(clustercenters) == 0
    covar = np.cov(clustercenters)
    cc = np.corrcoef(clustercenters)
    # when covar is 1x1 and thus clustercenters have only one feature
    if covar.ndim < 2:
        varianc = covar
    else:
        varianc = np.diagonal(covar)

    # check cc and variance is not NaNs
    if np.isnan(cc).any():
        markerlist = list(np.argsort(-clustercenters.ravel())[:markergoal])
        return markerlist

    thresh = 0.9
    increment = 0.1
    lowerthresh = 0.5

    vartemp = varianc.copy()
    cctemp = cc.copy()

    while True:
        hivar = np.argmax(vartemp)
        # print(hivar)
        if vartemp[hivar] == 0:
            if len(markerlist) == markergoal:
                # print('Just right')
                # print(markerlist)
                return markerlist
            else:
                if len(markerlist) > markergoal:
                    # print('Too many')
                    thresh = thresh - increment
                    if increment < 0.001:
                        # print('Truncating')
                        markerlist = markerlist[0:markergoal]
                        # print(markerlist)
                        return markerlist
                else:
                    if len(markerlist) < markergoal:
                        # print('Not enough')
                        increment = increment / 2
                        thresh = thresh + increment
                # reset to redo search
                # print('Resetting')
                vartemp = varianc.copy()
                cctemp = cc.copy()
                markerlist = []
        else:
            for i in range(0, covar.shape[0]):
                if abs(cctemp[i, hivar]) > thresh:
                    # zero out correlated features
                    vartemp[i] = 0
                    cctemp[i, :] = 0
                    cctemp[:, i] = 0
                    # print('Removing feature:')
                    # print(i)
            vartemp[hivar] = 0
            cctemp[hivar, :] = 0
            cctemp[:, hivar] = 0
            markerlist.append(hivar - 1)
            # print('Continuing')
            # print(markerlist)


def matchNShow_markers(clustercenters: np.ndarray, markerlist: List, features: List[str],
                       options: Dict) -> (Any, List[str]):
    '''
        get the markers to indicate what the respective clusters represent
    '''

    markers = [features[i] for i in markerlist]
    table = clustercenters[:, markerlist]

    if options.get("debug"):
        print(markerlist)
        print(table)

    return table, markers


def write_ometiff(im: IMGstruct, output_dir: Path, bestz: List, *argv):
    print('Writing out ometiffs for visualizations...')
    pcaimg = argv[0]
    pcaimg = pcaimg.reshape((pcaimg.shape[2], pcaimg.shape[0], pcaimg.shape[1]))
    pcaimg = pcaimg.astype(np.int32)

    superpixel = get_last2d(argv[1], bestz[0])
    superpixel = superpixel.astype(np.int32)
    superpixel = superpixel[np.newaxis, :, :]

    s = ["-channel_pca.ome.tiff", "-superpixel.ome.tiff"]
    f = [output_dir / (im.get_name() + s[0]), output_dir / (im.get_name() + s[1])]

    check_file_exist(f)
    writer = OmeTiffWriter(f[0])
    writer.save(pcaimg, channel_names=im.get_channel_labels(), image_name=im.get_name())
    writer = OmeTiffWriter(f[1])
    writer.save(superpixel, channel_names=im.get_channel_labels(), image_name=im.get_name())


def check_file_exist(paths: Path):
    for i in paths:
        if i.is_file():
            i.unlink()
        else:
            continue


def cell_cluster_IDs(filename: str, output_dir: Path, i: int, maskchs: List, options: Dict, *argv):
    allClusters = argv[0]
    for idx in range(1, len(argv)):
        allClusters = np.column_stack((allClusters, argv[idx]))
    # hard coded --> find a way to automate the naming
    if not options.get('skip_outlinePCA'):
        write_2_csv(list(['K-Means [Mean] Expression', 'K-Means [Covariance] Expression', 'K-Means [Total] Expression',
                          'K-Means [Mean-All-SubRegions] Expression', 'K-Means [Shape-Vectors]', 'K-Means [Texture]']),
                    allClusters,
                    filename + '-cell_cluster_' + maskchs[i], output_dir, options)
    else:
        write_2_csv(list(['K-Means [Mean] Expression', 'K-Means [Covariance] Expression', 'K-Means [Total] Expression',
                          'K-Means [Mean-All-SubRegions] Expression', 'K-Means [Texture]']), allClusters,
                    filename + '-cell_cluster_' + maskchs[i], output_dir, options)

    # write_2_csv(list(['K-Means [Mean] Expression', 'K-Means [Covariance] Expression', 'K-Means [Total] Expression',
    #                   'K-Means [Mean-All-SubRegions] Expression']), allClusters,
    #             filename + '-cell_cluster', output_dir, options)


def plot_img(cluster_im: np.ndarray, bestz: int, filename: str, output_dir: Path):
    cluster_im = get_last2d(cluster_im, bestz)
    plt.imshow(cluster_im, interpolation='nearest')
    plt.axis('off')
    # plt.show(block=False)
    f = output_dir / (filename)
    plt.savefig(f)
    plt.close()


def plot_imgs(filename: str, output_dir: Path, i: int, maskchs: List, options: Dict, *argv):
    plot_img(argv[0], 0, filename + '-Cluster_Means_' + maskchs[i] + '.png', output_dir)
    plot_img(argv[1], 0, filename + '-Cluster_Covar_' + maskchs[i] + '.png', output_dir)
    plot_img(argv[3], 0, filename + '-Cluster_Total_' + maskchs[i] + '.png', output_dir)

    if not options.get('skip_outlinePCA'):
        plot_img(argv[4], 0, filename + '-Cluster_Shape.png', output_dir)

    plot_img(argv[-1], 0, filename + '-Cluster_Texture.png', output_dir)
    plot_img(argv[2], 0, filename + '-Cluster_MeansAll.png', output_dir)

def make_legends(feature_names, feature_covar, feature_meanall, filename: str, output_dir: Path, i: int, options: Dict,
                 *argv):
    
    # make legend once
    if i == 0:


        print('Finding mean ALL cluster markers...')
        retmarkers = findmarkers(argv[3], options)
        table, markers = matchNShow_markers(argv[3], retmarkers, feature_meanall, options)
        write_2_csv(markers, table, filename + '-cluster_cell_meanALLCH_legend', output_dir, options)
        showlegend(markers, table, filename + '-cluster_cell_meanALLCH_legend.png', output_dir)

        if not options.get('skip_outlinePCA'):
            feature_shape = ['shapefeat ' + str(ff) for ff in range(0, argv[4].shape[1])]
            print('Finding cell shape cluster markers...')
            retmarkers = findmarkers(argv[4], options)
            table, markers = matchNShow_markers(argv[4], retmarkers, feature_shape, options)
            write_2_csv(markers, table, filename + '-cluster_cell_shape_legend', output_dir, options)
            showlegend(markers, table, filename + '-cluster_cell_shape_legend.png', output_dir)

        print('Finding cell texture cluster markers...')
        retmarkers = findmarkers(argv[-1][0], options)
        table, markers = matchNShow_markers(argv[-1][0], retmarkers, argv[-1][1], options)
        write_2_csv(markers, table, filename + '-cluster_cell_texture_legend', output_dir, options)
        showlegend(markers, table, filename + '-cluster_cell_texture_legend.png', output_dir)

    print('Legend for mask channel: ' + str(i))

    for j in range(len(argv)):

        # hard coded for argv idx and - psuedo switch -- might be a more efficient way
        if j == 0:
            print('Finding mean cluster markers...')
            retmarkers = findmarkers(argv[j], options)
            table, markers = matchNShow_markers(argv[j], retmarkers, feature_names, options)
            if i == 0:
                write_2_csv(markers, table, filename + '-cluster_cell_mean_legend', output_dir, options)
                showlegend(markers, table, filename + '-cluster_cell_mean_legend.png', output_dir)
            elif i == 1:
                write_2_csv(markers, table, filename + '-cluster_nuc_mean_legend', output_dir, options)
                showlegend(markers, table, filename + '-cluster_nuc_mean_legend.png', output_dir)
            elif i == 2:
                write_2_csv(markers, table, filename + '-cluster_cellsboundary_mean_legend', output_dir, options)
                showlegend(markers, table, filename + '-cluster_cellsboundary_mean_legend.png', output_dir)
            elif i == 3:
                write_2_csv(markers, table, filename + '-cluster_nucboundary_mean_legend', output_dir, options)
                showlegend(markers, table, filename + '-cluste_nucboundary_mean_legend.png', output_dir)

        elif j == 1:
            print('Finding covariance cluster markers...')
            retmarkers = findmarkers(argv[j], options)
            table, markers = matchNShow_markers(argv[j], retmarkers, feature_covar, options)

            if i == 0:
                write_2_csv(markers, table, filename + '-cluster_cell_covariance_legend', output_dir, options)
                showlegend(markers, table, filename + '-cluster_cell_covariance_legend.png', output_dir)
            elif i == 1:
                write_2_csv(markers, table, filename + '-cluster_nuc_covariance_legend', output_dir, options)
                showlegend(markers, table, filename + '-cluster_nuc_covariance_legend.png', output_dir)
            elif i == 2:
                write_2_csv(markers, table, filename + '-cluster_cellsboundary_covariance_legend', output_dir, options)
                showlegend(markers, table, filename + '-cluster_cellsboundary_covariance_legend.png', output_dir)
            elif i == 3:
                write_2_csv(markers, table, filename + '-cluster_nucboundary_covariance_legend', output_dir, options)
                showlegend(markers, table, filename + '-cluste_nucboundary_covariance_legend.png', output_dir)

        elif j == 2:
            print('Finding total cluster markers...')
            retmarkers = findmarkers(argv[j], options)
            table, markers = matchNShow_markers(argv[j], retmarkers, feature_names, options)

            if i == 0:
                write_2_csv(markers, table, filename + '-cluster_cell_total_legend', output_dir, options)
                showlegend(markers, table, filename + '-cluster_cell_total_legend.png', output_dir)
            elif i == 1:
                write_2_csv(markers, table, filename + '-cluster_nuc_total_legend', output_dir, options)
                showlegend(markers, table, filename + '-cluster_nuc_total_legend.png', output_dir)
            elif i == 2:
                write_2_csv(markers, table, filename + '-cluster_cellsboundary_total_legend', output_dir, options)
                showlegend(markers, table, filename + '-cluster_cellsboundary_total_legend.png', output_dir)
            elif i == 3:
                write_2_csv(markers, table, filename + '-cluster_nucboundary_total_legend', output_dir, options)
                showlegend(markers, table, filename + '-cluster_nucboundary_total_legend.png', output_dir)


def save_all(filename: str, im: IMGstruct, mask: MaskStruct, output_dir: Path, options: Dict, *argv):
    # hard coded for now
    print('Writing to csv all matrices...')
    mean_vector = argv[0]
    covar_matrix = argv[1]
    total_vector = argv[2]

    if not options.get('skip_outlinePCA'):
        outline_vectors = argv[3]
        write_2_file(outline_vectors, filename + '-cell_shape', im, output_dir, options)

    write_2_file(mean_vector[0, -1, :, :, 0], filename + '-cell_channel_meanAllChannels', im, output_dir, options)
    # write_2_file(texture_v[0, -1, :, :, 0], filename + '-cell_channel_textures', im, output_dir, options)

    for i in range(len(mask.channel_labels)):
        write_2_file(mean_vector[0, i, :, :, 0], filename + mask.get_channel_labels()[i] + 'channel_mean', im,
                     output_dir, options)
        write_2_file(covar_matrix[0, i, :, :, :], filename + mask.get_channel_labels()[i] + '-cell_channel_covar', im,
                     output_dir, options)
        write_2_file(total_vector[0, i, :, :, 0], filename + mask.get_channel_labels()[i] + '-cell_channel_total', im,
                     output_dir, options)


def cell_analysis(im: IMGstruct, mask: MaskStruct, filename: str, bestz: int, output_dir: Path, seg_n: int,
                  options: Dict, *argv):
    '''
        cluster and statisical analysis done on cell:
        clusters/maps and makes a legend out of the most promient channels and writes them to a csv
    '''
    stime = time.monotonic() if options.get("debug") else None
    # hard coded for now
    mean_vector = argv[0]
    covar_matrix = argv[1]
    total_vector = argv[2]

    if not options.get('skip_outlinePCA'):
        shape_vectors = argv[3]
    else:
        texture_vectors = argv[3][0]
        texture_channels = argv[3][1]

    texture_vectors = argv[4][0]
    texture_channels = argv[4][1]

    # get channel labels
    maskchs = mask.get_channel_labels()
    feature_names = im.get_channel_labels()
    feature_covar = options.get('channel_label_combo')
    feature_meanall = feature_names + feature_names + feature_names + feature_names

    # features only clustered once
    meanAll_vector_f = cell_cluster_format(mean_vector, -1, options)
    clustercells_uvall, clustercells_uvallcenters = cell_cluster(meanAll_vector_f,
                                                                 options)  # -1 means use all segmentations

    if options.get('skip_texture'):
        options['texture_flag'] = True

    texture_matrix = cell_cluster_format(texture_vectors, -1, options)
    clustercells_texture, clustercells_texturecenters = cell_cluster(texture_matrix, options)

    cluster_cell_imguall = cell_map(mask, clustercells_uvall, seg_n, options)  # 0=use first segmentation to map
    cluster_cell_texture = cell_map(mask, clustercells_texture, seg_n, options)

    if not options.get('skip_outlinePCA'):
        clustercells_shapevectors, shapeclcenters = shape_cluster(shape_vectors, options)
        clustercells_shape = cell_map(mask, clustercells_shapevectors, seg_n, options)

    # for each channel in the mask
    for i in range(len(maskchs)):
        seg_n = mask.get_labels(maskchs[i])

        # format the feature arrays accordingly
        mean_vector_f = cell_cluster_format(mean_vector, seg_n, options)
        covar_matrix_f = cell_cluster_format(covar_matrix, seg_n, options)
        total_vector_f = cell_cluster_format(total_vector, seg_n, options)

        # cluster by mean and covar using just cell segmentation mask
        print('Clustering cells and getting back labels and centers...')
        clustercells_uv, clustercells_uvcenters = cell_cluster(mean_vector_f, options)
        clustercells_cov, clustercells_covcenters = cell_cluster(covar_matrix_f, options)
        clustercells_total, clustercells_totalcenters = cell_cluster(total_vector_f, options)

        # map back to the mask segmentation of indexed cell region
        print('Mapping cell index in segmented mask to cluster IDs...')
        cluster_cell_imgu = cell_map(mask, clustercells_uv, seg_n, options)
        cluster_cell_imgcov = cell_map(mask, clustercells_cov, seg_n, options)
        cluster_cell_imgtotal = cell_map(mask, clustercells_total, seg_n, options)

        print('Getting markers that separate clusters to make legend...')
        if not options.get('skip_outlinePCA'):
            # get markers for each respective cluster & then save the legend/markers
            make_legends(feature_names, feature_covar, feature_meanall, filename, output_dir, i, options,
                         clustercells_uvcenters, clustercells_covcenters,
                         clustercells_totalcenters,
                         clustercells_uvallcenters,
                         shapeclcenters, [clustercells_texturecenters, texture_channels])
            # save all clusterings to one csv
            print('Writing out all cell cluster IDs for all cell clusterings...')
            cell_cluster_IDs(filename, output_dir, i, maskchs, options, clustercells_uv, clustercells_cov,
                             clustercells_total,
                             clustercells_uvall,
                             clustercells_shapevectors, clustercells_texture)
            # plots the cluster imgs for the best z plane
            print('Saving pngs of cluster plots by best focal plane...')
            plot_imgs(filename, output_dir, i, maskchs, options, cluster_cell_imgu[bestz], cluster_cell_imgcov[bestz],
                      cluster_cell_imguall[bestz],
                      cluster_cell_imgtotal[bestz], clustercells_shape[bestz], cluster_cell_texture[bestz])
        else:
            make_legends(feature_names, feature_covar, feature_meanall, filename, output_dir, i, options,
                         clustercells_uvcenters, clustercells_covcenters,
                         clustercells_totalcenters,
                         clustercells_uvallcenters, [clustercells_texturecenters, texture_channels])
            # save all clusterings to one csv
            print('Writing out all cell cluster IDs for all cell clusterings...')
            cell_cluster_IDs(filename, output_dir, i, maskchs, options, clustercells_uv, clustercells_cov,
                             clustercells_total,
                             clustercells_uvall, clustercells_texture)
            # plots the cluster imgs for the best z plane
            print('Saving pngs of cluster plots by best focal plane...')
            plot_imgs(filename, output_dir, i, options, cluster_cell_imgu[bestz], cluster_cell_imgcov[bestz],
                      cluster_cell_imguall[bestz],
                      cluster_cell_imgtotal[bestz], cluster_cell_texture[bestz])

    if options.get("debug"): print('Elapsed time for cluster img saving: ', time.monotonic() - stime)


def make_DOT(mc, fc, coeffs, ll):
    pass


def powerset(iterable: List[int]):
    '''
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def read_options(options_path: Path) -> Dict[str, Union[int, str]]:
    # read in options
    options = {}
    with open(options_path) as f:
        for line in f:
            split = line.split()
            if len(split) < 3:
                (key, value) = split
                value = is_number(value)
                options[key] = value
            else:
                key = split[0]
                value = split[1:]
                for i in range(len(value)):
                    value[i] = is_number(value[i])
                options[key] = value
    return options


def is_number(val: Any) -> Any:
    try:
        val = int(val)
    except ValueError:
        try:
            val = float(val)
        except ValueError:
            val = val
    return val


def check_output_dir(path: Path, options: Dict):
    if Path(path).is_dir():
        if options.get("debug"): print('Output directory exists')
    else:
        path.mkdir()
        if options.get("debug"): print('Output directory created')


def showlegend(markernames: List[str], markertable: np.ndarray, outputfile: str, output_dir: Path):
    mmin = []
    mmax = []
    for k in range(0, len(markernames)):
        mmin.append(min(markertable[:, k]))
        mmax.append(max(markertable[:, k]))
    for i in range(0, len(markertable)):
        tplot = [(markertable[i, k] - mmin[k]) / (mmax[k] - mmin[k])
                 for k in range(0, len(markernames))]
        plt.plot(tplot, label=['cluster ' + str(i)])
    plt.xlabel('-'.join(markernames))
    plt.ylabel('Relative value')
    plt.legend()
    plt.tick_params(axis="x", which="both", bottom=False, top=False)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_ticks([])
    # plt.show(block=False)
    f = output_dir / outputfile
    plt.savefig(f)
    plt.close()


def find_locations(arr: np.ndarray) -> (np.ndarray, List[np.ndarray]):
    coords = np.indices(arr.shape).reshape(arr.ndim, arr.size)
    arr = arr.ravel()
    order = np.argsort(arr)
    arr = arr[order]
    coords = coords[:, order]
    locs = np.flatnonzero(np.diff(arr, prepend=arr[0] - 1))
    return arr[locs], np.split(coords, locs[1:], axis=1)


def get_last2d(data: np.ndarray, bestz: int) -> np.ndarray:
    if data.ndim <= 2:
        return data
    slc = [0] * (data.ndim - 3)
    slc += [bestz, slice(None), slice(None)]
    return data[tuple(slc)]


def summary(im, total_cells: List, img_files: Path, output_dir: Path, options: Dict):
    '''
        Write out summary csv of full image analysis (combination of all tiles)
    '''
    channel_n = im.get_channel_labels().copy()
    channel_n = list(map(lambda x: x + ': SNR', channel_n))
    img_files = list(map(lambda x: x.name, img_files))
    # img_files = img_files[0:3]

    snr_paths = output_dir / '*-SNR.csv'
    snr_files = get_paths(snr_paths)

    for i in range(0, len(snr_files)):
        df1 = pd.read_csv(snr_files[i])

        np1 = df1.to_numpy()
        zscore = np1[0, 1:]
        otsu = np1[1, 1:]

        if i == 0:
            f1 = zscore
            f2 = otsu
        else:
            f1 = np.vstack((f1, zscore))
            f2 = np.vstack((f2, otsu))

    if i == 0:
        f1 = f1.reshape(1, -1)
        f2 = f2.reshape(1, -1)

    df1 = pd.DataFrame(f1, columns=channel_n)
    df2 = pd.DataFrame(f2, columns=channel_n)

    df3 = pd.DataFrame({'Filename': img_files, 'Total Cells': total_cells})

    df1 = pd.concat([df3, df1], axis=1, sort=False)
    df2 = pd.concat([df3, df2], axis=1, sort=False)

    df1.to_csv(output_dir / 'summary_zscore.csv', index=False)
    df2.to_csv(output_dir / 'summary_otsu.csv', index=False)


def check_shape(im, mask):
    # put in check here for matching dims

    return im.get_data().shape[4] != mask.get_data().shape[4] or im.get_data().shape[5] != mask.get_data().shape[5]


def reallocate_parallel(im, mask, ichan, options):
    pass


def reallocate_and_merge_intensities(im, mask, optional_img_file, options):
    # check if image needs to be resized to match mask
    if check_shape(im, mask):
        s, t, c, z, y, x = im.get_data().shape

        # allocate memory for reallocated im
        newim = np.zeros((s, t, c, z, mask.get_data().shape[4], mask.get_data().shape[5]))

        ROI = mask.get_data()[0, 0, 0, 0, :, :]  # assume chan 0 is the cell mask
        IMSimg = im.get_data()[0, 0, 0, 0, :, :]

        # do findpixelpixelfraction and getrelevantpixel index once
        print('START findpixelfraction...')
        X, A, cellArea, reducedsize = findpixelfractions(ROI.reshape(-1), ROI.shape, IMSimg.shape, c)
        # print(cellArea)
        # exit()

        print('END findpixelfraction...')
        newim[0, 0, :, 0, :, :] = reallo(im, ROI, X, A, cellArea, reducedsize, options)

        if z != 1:
            newim = np.repeat(newim, z, axis=3)
        # for ichan in range(0, im.data.shape[2]):
        #     # allocate portions of the IMS to corresponding areas in the mask
        #     newim[0, 0, ichan, 0, :, :] = reallocateIMS(im, ROI, ichan, X, A, cellArea, reducedsize, options)

        im.set_data(newim)
        newim = None

    # check if additional image needs to be stacked with it
    if optional_img_file:
        # img_path = options_img_dir / im.name
        im2 = IMGstruct(optional_img_file, options)
        stacked_img = np.concatenate((im.get_data(), im2.get_data()), axis=2)
        channel_list = im.get_channel_labels()
        channel_list.extend(im2.get_channel_labels())
        im2.quit()

        im.set_data(stacked_img)
        im.set_channel_labels(channel_list)

    # prune for NaNs
    im_prune = im.get_data()
    nan_find = np.isnan(im_prune)
    im_prune[nan_find] = 0
    im.set_data(im_prune)


# def generate_fake_stackimg(im, mask, opt_img_file, options):
#     #not for general purposes
#
#     c = im.get_data().shape[2]
#     z, y, x = mask.get_data().shape[3], mask.get_data().shape[4], mask.get_data().shape[5]
#
#     im2 = IMGstruct(opt_img_file, options)
#     channel_list = im.get_channel_labels()
#     channel_list.extend(im2.get_channel_labels())
#
#     c2, z2 = im2.get_data().shape[2], im2.get_data().shape[3]
#
#     im2.quit()
#
#     print('Start of fake image creation')
#     fake = np.random.choice([0, 100, 1000, 10000, 100000], size=(1, 1, c+c2, z+z2, y, x), p=[0.9, 0.025, 0.025, 0.025, 0.025])
#     im.set_data(fake)
#     im.set_channel_labels(channel_list)
#     print('END')

def recluster(output_dir: Path, im: IMGstruct, options: Dict):
    filename = 'Recluster'

    # features
    meanV = '*-cell_channel_mean.csv'
    covar = '*-cell_channel_covar.csv'
    totalV = '*-cell_channel_total.csv'
    # meanC = '*-cell_channel_meanAllChannels.csv'
    shapeV = '*-cell_shape.csv'

    # read in and concatenate all feature csvs
    meanAll = get_paths(output_dir / meanV)
    covarAll = get_paths(output_dir / covar)
    totalAll = get_paths(output_dir / totalV)
    # meanCAll = get_paths(output_dir / meanC)
    shapeAll = get_paths(output_dir / shapeV)

    for i in range(0, len(meanAll)):
        meanAll_read = pd.read_csv(meanAll[i])
        covarAll_read = pd.read_csv(covarAll[i])
        totalAll_read = pd.read_csv(totalAll[i])
        # meanCAll_read = pd.read_csv(meanCAll[i])
        shapeAll_read = pd.read_csv(shapeAll[i])

        if i == 0:
            meanAll_pd = meanAll_read
            covarAll_pd = covarAll_read
            totalAll_pd = totalAll_read
            # meanCALL_pd = meanCAll_read
            shapeAll_pd = shapeAll_read
        else:
            meanAll_pd = pd.concat([meanAll_pd, meanAll_read], axis=1, sort=False)
            covarAll_pd = pd.concat([covarAll_pd, covarAll_read], axis=1, sort=False)
            totalAll_pd = pd.concat([totalAll_pd, totalAll_read], axis=1, sort=False)
            # meanCALL_pd = pd.concat([meanCALL_pd, meanCAll_read], axis=1, sort=False)
            shapeAll_pd = pd.concat([shapeAll_pd, shapeAll_read], axis=1, sort=False)

    meanAll_np = meanAll_pd.to_numpy()
    covarAll_np = covarAll_pd.to_numpy()
    totalAll_np = totalAll_pd.to_numpy()
    # meanCALL_np = meanCALL_pd.to_numpy()
    shapeAll_np = shapeAll_pd.to_numpy()

    print('Reclustering cells and getting back the labels and centers...')
    clustercells_uv, clustercells_uvcenters = cell_cluster(meanAll_np, options)
    clustercells_cov, clustercells_covcenters = cell_cluster(covarAll_np, options)
    clustercells_total, clustercells_totalcenters = cell_cluster(totalAll_np, options)
    clustercells_uvall, clustercells_uvallcenters = cell_cluster(meanAll_np, options)
    clustercells_shapevectors, shapeclcenters = shape_cluster(shapeAll_np, options)

    print('Making legend for the recluster...')
    make_legends(im, filename, output_dir, options, clustercells_uvcenters, clustercells_covcenters,
                 clustercells_totalcenters, clustercells_uvallcenters,
                 shapeclcenters)

    print('Writing out all cell cluster IDs for recluster cells...')
    cell_cluster_IDs(filename, output_dir, options, clustercells_uv, clustercells_cov, clustercells_total,
                     clustercells_uvall,
                     clustercells_shapevectors)


def quality_control(mask: MaskStruct, img: IMGstruct, ROI_coords: List, options: Dict):
    # best z +- from options
    set_zdims(mask, img, options)

    # find cells on edge
    find_edge_cells(mask)

    #

    # normalize bg intensities
    if options.get('normalize_bg'):
        normalize_background(img, ROI_coords)


def normalize_background(im, ROI_coords):
    # pass
    img = im.get_data()

    for i in range(img.shape[2]):
        img_ch = img[0, 0, i, 0, :, :]
        bg_img_ch = img_ch[ROI_coords[0][0][0, :], ROI_coords[0][0][1, :]]

        avg = np.sum(bg_img_ch) / ROI_coords[0][0].shape[1]

        if avg == 0:
            continue

        img[0, 0, i, 0, :, :] = img_ch / avg

    im.set_data(img)


def set_zdims(mask: MaskStruct, img: IMGstruct, options: Dict):
    bound = options.get("zslices")
    bestz = mask.get_bestz()
    z = mask.get_data().shape[3]

    lower_bound = bestz[0] - bound
    upper_bound = bestz[0] + bound + 1
    if lower_bound < 0 or upper_bound > z:
        print('zslice bound is invalid. Please reset and run SPRM')
        exit(0)
    else:
        new_mask = mask.get_data()[:, :, :, lower_bound:upper_bound, :, :]
        new_img = img.get_data()[:, :, :, lower_bound:upper_bound, :, :]

        mask.set_data(new_mask)
        mask.set_bestz([0])  # set best z back to 0 since we deleted some zstacks
        img.set_data(new_img)


def find_edge_cells(mask):
    # 3D case
    channels = mask.get_data().shape[2]
    bestz = mask.get_bestz()[0]
    data = mask.get_data()[0, 0, :, bestz, :, :]
    border = []
    for i in range(0, channels):
        border += list(data[i, 0, :-1])  # Top row (left to right), not the last element.
        border += list(data[i, :-1, -1])  # Right column (top to bottom), not the last element.
        border += list(data[i, -1, :0:-1])  # Bottom row (right to left), not the last element.
        border += list(data[i, ::-1, 0])  # Left column (bottom to top), all elements element.

    border = np.unique(border).tolist()
    mask.set_edge_cells(border)

    sMask = mask.get_data()[0, 0, 0, :, :, :]
    unique = np.unique(sMask)[1:]
    interiorCells = [i for i in unique if i not in border and i not in mask.get_bad_cells()]
    # interiorCells = [i for i in unique if i not in mask.get_bad_cells()]
    mask.set_interior_cells(interiorCells)

def glcm(im, mask, bestz, output_dir, cell_total, filename, options, angle, distances):
    '''
    By: Young Je Lee
    '''

    colIndex = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    texture_all = pd.DataFrame()
    for i in range(int(len(mask.channel_labels) / 2)):  # latter ones are edge
        texture = pd.DataFrame()  # Cell, Nuclei
        for distance in distances:
            for j in range(len(im.channel_labels)):  # For each channel
                print("current calculation:",im.channel_labels[j]+"_"+str(distance)+"_"+mask.channel_labels[i])
                originalImg = im.data[0, 0, j, bestz[0], :, :]
                originalMask = mask.data[0, 0, i, bestz[0], :, :]
                tex = pd.DataFrame()
                for ls in range(len(colIndex)):
                    tex.insert(len(tex.columns), str(im.channel_labels[j] + ":" + colIndex[ls]+":"+str(distance)+":"+mask.channel_labels[i]), 0)
                for idx in range(cell_total[0] + 1):  # For each cell
                    img =originalImg.copy()
                    interiormask = originalMask.copy()
                    interiormask=(interiormask==idx)                    
                    img=np.multiply(interiormask,img)
                    img=uint16_2_uint8(img)
                    result = greycomatrix(img.astype(np.uint8), [distance], [angle],levels=256)  # Calculate GLCM
                    result = result[1:, 1:]  # Remove background influence by delete first row & column
                    props = []
                    for ls in range(len(colIndex)):  # Get properties
                        props.append(greycoprops(result, colIndex[ls]).flatten()[0])
                    tex.loc[idx] = props
                if len(texture) == 0:
                    texture = tex
                else:
                    texture = pd.concat([texture, tex], axis=1)
            texture = texture.drop(0)
            texture.index.name = 'ID'
            texture.to_csv(output_dir / (filename + '_' + mask.channel_labels[i] + '_' + str(distance) + '_texture.csv'))
        if len(texture_all) == 0:
            texture_all = texture
        else:
            texture_all = pd.concat([texture_all, texture], axis=1)
    texture_featureNames=list(texture_all.columns)
    texture_all = texture_all.to_numpy()
    texture_all=np.reshape(texture_all,(1,int(len(mask.channel_labels) / 2),cell_total[0],len(im.channel_labels),len(colIndex)*len(distances)))
    return texture_all, texture_featureNames


def uint16_2_uint8(uint16matrix):
    maxvalue = np.max(uint16matrix)
    if maxvalue==0:
        return uint16matrix
    return uint16matrix*(255 / maxvalue) 

def glcmProcedure(im, mask, bestz, output_dir, cell_total, filename, options):
    angle = options.get('glcm_angles')
    distances = options.get('glcm_distances')
    angle=''.join(angle)[1:-1].split(',')
    distances=''.join(distances)[1:-1].split(',')
    angle = [int(i) for i in angle][0] #Only supports 0 for now
    distances= [int(i) for i in distances]
    texture,texture_featureNames = glcm(im, mask, bestz, output_dir, cell_total, filename, options, angle, distances)
    print("GLCM calculations completed")
    return [texture, texture_featureNames]

