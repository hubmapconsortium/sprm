import math
import re
import time
from itertools import chain, combinations, product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from .outlinePCA import shape_cluster

"""
Companion to SPRM.py
Package functions that are integral to running main script
Author:    Ted Zhang & Robert F. Murphy
01/21/2020 - 07/19/2020
Version: 0.56
"""


INTEGER_PATTERN = re.compile(r'(\d+)')
FILENAMES_TO_IGNORE = frozenset({'.DS_Store'})


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
        else:
            if options.get("debug"): print('Metadata found in input image')

        return img

    def read_data(self, options):
        data = self.img.data
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


class MaskStruct(IMGstruct):

    def __init__(self, path: Path, options):
        super().__init__(path, options)
        self.bestz = self.get_bestz()

    def get_labels(self, label):
        return self.channel_labels.index(label)

    def set_bestz(self, z):
        self.bestz = z

    def get_bestz(self):
        return self.bestz

    def read_data(self, options):
        bestz = None
        data = self.img.data
        dims = data.shape
        # s,t,c,z,y,x = dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]
        check = data[:, :, :, 0, :, :]
        check_sum = np.sum(check)
        if check_sum == 0: #assumes the best z is not the first slice
            print('Duplicating best z to all z dimensions...')
            for i in range(0, data.shape[3]):
                x = data[:, :, :, i, :, :]
                y = np.sum(x)
                #                print(x)
                #                print(y)
                if y > 0:
                    bestz = i
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
            bestz=0

        #set bestz
        self.set_bestz(bestz)


        return data

def calculations(coord, im: IMGstruct, t: int, i: int) -> (np.ndarray, np.ndarray, np.ndarray):
    '''
        Returns covariance matrix, mean vector, and total vector
    '''

    if i == 0: print('Performing statistical analyses on ROIs...')

    z, y, x = coord[0], coord[1], coord[2]

    temp = im.get_data()

    channel_all_mask = temp[0, t, :, z, y, x]
    ROI = np.transpose(channel_all_mask)

    cov_m = np.cov(ROI)
    mu_v = np.reshape(np.mean(ROI, axis=1), (ROI.shape[0], 1))
    total = np.reshape(np.sum(ROI, axis=1), (ROI.shape[0], 1))

    if not cov_m.shape:
        cov_m = np.array([cov_m])
        # print(cov_m)
        # print(cov_m.shape)

    return cov_m, mu_v, total


def cell_cluster(cell_matrix: np.ndarray, segnum: int, options: Dict) -> (np.ndarray, np.ndarray):
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
        temp_matrix = np.zeros([cell_matrix.shape[1], cell_matrix.shape[2], cell_matrix.shape[3], cell_matrix.shape[0]])
        for i in range(0, cell_matrix.shape[0]):
            temp_matrix[:, :, :, i] = cell_matrix[i, :, :, :]
        cell_matrix = temp_matrix
        cell_matrix = cell_matrix.reshape(
            (cell_matrix.shape[0], cell_matrix.shape[1] * cell_matrix.shape[2] * cell_matrix.shape[3]))
    # kmeans clustering
    print('Clustering cells...')
    #check of clusters vs. n_sample wanted
    num_cellclusters = options.get("num_cellclusters")
    if num_cellclusters > cell_matrix.shape[0]:
        print('reducing cell clusters to ', cell_matrix.shape[0])
        num_cellclusters = cell_matrix.shape[0]
    cellbycluster = KMeans(n_clusters= num_cellclusters, random_state=0).fit(cell_matrix)
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
    temp = mask_img.copy()
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


def mask_img(mask: MaskStruct, j: int) -> (np.ndarray, np.ndarray):
    '''
    returns: a 3D matrix that represents the jth segmented image
    '''
    print('Getting indexed mask from ' + mask.get_channel_labels()[j] + ' channel')
    sMask = mask.get_data()

    # print(sMask.shape)
    sMask = sMask[0, 0, j, :, :, :]
    unique = np.unique(sMask)

    return sMask, unique


def get_masked_imgs(labeled_mask: np.ndarray, maskIDs: np.ndarray) -> List[np.ndarray]:
    '''
        Returns the masked image as a set of coordinates
    '''
    print('Getting coordinates that correspond with ROIs from indexed mask...')
    masked_imgs_coord = []
    # need to implement a more efficient algo
    for i in range(1, len(maskIDs)):
        coor = np.where(labeled_mask == maskIDs[i])
        masked_imgs_coord.append(coor)

    return masked_imgs_coord


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
    f = output_dir / (s+'.csv')

    #column header for indices
    df.index.name = 'ID'
    df.to_csv(f, header=header)

def write_cell_polygs(polyg_list: List[np.ndarray], filename: str, output_dir: Path, options: Dict):
    coord_pairs = []
    for i in range(0, len(polyg_list)):
        tlist = str([[round(i, 4), round(j, 4)] for i, j in zip(polyg_list[i][:, 0], polyg_list[i][:, 1])])
        coord_pairs.append(tlist)

    df = pd.DataFrame({0:coord_pairs}, index=list(range(1,len(coord_pairs)+1)))
    if options.get("debug"): print(df)
    f = output_dir / (filename+'-cell_polygons_spatial.csv')
    df.index.name = 'ID'
    df.to_csv(f, header=['Shape'])

def build_matrix(im: IMGstruct, mask: MaskStruct, masked_imgs_coord: List[np.ndarray], j: int,
                 omatrix: np.ndarray) -> np.ndarray:
    if j == 0:
        return np.zeros(
            (im.get_data().shape[1], mask.get_data().shape[2], len(masked_imgs_coord), im.get_data().shape[2], im.get_data().shape[2]))
    else:
        return omatrix


def build_vector(im: IMGstruct, mask: MaskStruct, masked_imgs_coord: List[np.ndarray], j: int,
                 omatrix: np.ndarray) -> np.ndarray:
    if j == 0:
        return np.zeros((im.get_data().shape[1], mask.get_data().shape[2], len(masked_imgs_coord), im.get_data().shape[2], 1))
    else:
        return omatrix


def clusterchannels(im: IMGstruct, options: Dict) -> np.ndarray:
    '''
        cluster all channels using PCA
    '''
    print('Dimensionality Reduction of image channels...')
    if options.get("debug"): print('Image dimensions before reduction: ', im.get_data().shape)
    pca_channels = PCA(n_components=options.get("n_components"), svd_solver='full')
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
    plt.savefig(f, box_inches='tight')
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
    snr_channels = np.where(sd == 0, 0, m/sd)
    # snr_channels = snr_channels[np.newaxis, :]
    # snr_channels = snr_channels.tolist()

    # define otsu threshold
    for i in range(0, channvals.shape[0]):
        img_2D = channvals[i, :, :, :]
        img_2D = img_2D.reshape((img_2D.shape[0] * img_2D.shape[1], img_2D.shape[2]))
        
        try:
            thresh = threshold_otsu(img_2D)
        except ValueError:
            thresh = 0
            
        above_th = img_2D > thresh
        below_th = img_2D < thresh
        
        mean_a = np.mean(img_2D[above_th])
        mean_b = np.mean(img_2D[below_th])

        # all pixels are in the foreground
        if mean_b == 0:
            fbr = 1
        else:
            fbr = mean_a/mean_b

        # take ratio above thresh / below thresh
        channelist.append(fbr)
        
    channelist = np.asarray(channelist)
    
    snrs = np.stack([snr_channels, channelist])
    
    #check for nans
    snrs[np.isnan(snrs)] = 0
    
    #add index identifier
    row_index = 1
    options['row_index_names'] = ['Z-Score', 'Otsu']
    
    # write out 2 rows
    write_2_csv(im.get_channel_labels(), snrs, filename +'-SNR', output_dir, options)
    
    #turn index boolean off
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
        
    covar = np.cov(clustercenters)
    cc = np.corrcoef(clustercenters)
    
    if covar.ndim < 2:
        variance = covar
    else:
        varianc = np.diagonal(covar)
        
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
            markerlist.append(hivar)
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


def write_ometiff(im: IMGstruct, output_dir: Path, bestz: int, *argv):
    print('Writing out ometiffs for visualizations...')
    pcaimg = argv[0]
    pcaimg = pcaimg.reshape((pcaimg.shape[2], pcaimg.shape[0], pcaimg.shape[1]))
    pcaimg = pcaimg.astype(np.int32)
    
    superpixel = get_last2d(argv[1], bestz)
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


def cell_cluster_IDs(filename: str, output_dir: Path, options: Dict, *argv):
    allClusters = argv[0]
    for idx in range(1, len(argv)):
        allClusters = np.column_stack((allClusters, argv[idx]))
    # hard coded --> find a way to automate the naming
    write_2_csv(list(['K-Means [Mean] Expression', 'K-Means [Covariance] Expression', 'K-Means [Total] Expression', 'K-Means [Mean-All-SubRegions] Expression', 'K-Means [Shape-Vectors]']), allClusters,
                filename + '-cell_cluster', output_dir, options)


def plot_img(cluster_im: np.ndarray, bestz: int, filename: str, output_dir: Path):
    cluster_im = get_last2d(cluster_im, bestz)
    plt.imshow(cluster_im, interpolation='nearest')
    plt.axis('off')
    # plt.show(block=False)
    f = output_dir / (filename)
    plt.savefig(f, box_inches='tight')
    plt.close()


def plot_imgs(filename: str, output_dir: Path, bestz: int, *argv):
    plot_img(argv[0], bestz, filename + '-ClusterByMeansPerCell.png', output_dir)
    plot_img(argv[1], bestz, filename + '-ClusterByCovarPerCell.png', output_dir)
    plot_img(argv[2], bestz, filename + '-ClusterByMeansAllMasks.png', output_dir)
    plot_img(argv[3], bestz, filename + '-ClusterByTotalPerCell.png', output_dir)
    plot_img(argv[4], bestz, filename + '-ClusterByShape.png', output_dir)


def make_legends(im: IMGstruct, filename: str, output_dir: Path, options: Dict, *argv):
    feature_names = im.get_channel_labels()
    feature_covar = options.get('channel_label_combo')
    feature_meanall = feature_names + feature_names + feature_names + feature_names
    feature_shape = ['shapefeat ' + str(ff) for ff in range(0, argv[4].shape[1])]

    print('Finding cell mean cluster markers...')
    retmarkers = findmarkers(argv[0], options)
    table, markers = matchNShow_markers(argv[0], retmarkers, feature_names, options)
    write_2_csv(markers, table, filename + '-clustercells_cellmean_legend', output_dir, options)
    showlegend(markers, table, filename + '-clustercells_cellmean_legend.png', output_dir)

    print('Finding cell covariance cluster markers...')
    retmarkers = findmarkers(argv[1], options)
    table, markers = matchNShow_markers(argv[1], retmarkers, feature_covar, options)
    write_2_csv(markers, table, filename + '-clustercells_cellcovariance_legend', output_dir, options)
    showlegend(markers, table, filename + '-clustercells_cellcovariance_legend.png', output_dir)

    print('Finding cell total cluster markers...')
    retmarkers = findmarkers(argv[2], options)
    table, markers = matchNShow_markers(argv[2], retmarkers, feature_names, options)
    write_2_csv(markers, table, filename + '-clustercells_celltotal_legend', output_dir, options)
    showlegend(markers, table, filename + '-clustercells_celltotal_legend.png', output_dir)

    print('Finding cell mean ALL cluster markers...')
    retmarkers = findmarkers(argv[3], options)
    table, markers = matchNShow_markers(argv[3], retmarkers, feature_meanall, options)
    write_2_csv(markers, table, filename + '-clustercells_cellmeanALL_legend', output_dir, options)
    showlegend(markers, table, filename + '-clustercells_cellmeanALL_legend.png', output_dir)

    print('Finding cell shape cluster markers...')
    retmarkers = findmarkers(argv[4], options)
    table, markers = matchNShow_markers(argv[4], retmarkers, feature_shape, options)
    write_2_csv(markers, table, filename + '-clustercells_cellshape_legend', output_dir, options)
    showlegend(markers, table, filename + '-clustercells_cellshape_legend.png', output_dir)


def save_all(filename: str, im: IMGstruct, seg_n: int, output_dir: Path, options: Dict, *argv):
    # hard coded for now
    print('Writing to csv all matrices...')
    mean_vector = argv[0]
    covar_matrix = argv[1]
    total_vector = argv[2]
    outline_vectors = argv[3]
    write_2_file(mean_vector[0, seg_n, :, :, 0], filename + '-cell_channel_mean', im, output_dir, options)
    write_2_file(covar_matrix[0, seg_n, :, :, :], filename + '-cell_channel_covar', im, output_dir, options)
    write_2_file(total_vector[0, seg_n, :, :, 0], filename + '-cell_channel_total', im, output_dir, options)
    write_2_file(outline_vectors, filename + '-cell_shape', im, output_dir, options)


def cell_analysis(im: IMGstruct, mask: MaskStruct, filename: str, bestz: int, seg_n: int, output_dir: Path, options: Dict, *argv):
    '''
        cluster and statisical analysis done on cell:
        clusters/maps and makes a legend out of the most promient channels and writes them to a csv
    '''
    stime = time.monotonic() if options.get("debug") else None
    # hard coded for now
    mean_vector = argv[0]
    covar_matrix = argv[1]
    total_vector = argv[2]
    shape_vectors = argv[3]
    # cluster by mean and covar using just whole cell masks
    print('Clustering cells and getting back labels and centers...')
    clustercells_uv, clustercells_uvcenters = cell_cluster(mean_vector, seg_n, options)
    clustercells_cov, clustercells_covcenters = cell_cluster(covar_matrix, seg_n, options)
    clustercells_total, clustercells_totalcenters = cell_cluster(total_vector, seg_n, options)
    clustercells_uvall, clustercells_uvallcenters = cell_cluster(mean_vector, -1,
                                                                 options)  # -1 means use all segmentations
    clustercells_shapevectors, shapeclcenters = shape_cluster(shape_vectors, options)
    # map back to the mask segmentation of indexed cell region
    print('Mapping cell index in segmented mask to cluster IDs...')
    cluster_cell_imgu = cell_map(mask, clustercells_uv, seg_n, options)
    cluster_cell_imgcov = cell_map(mask, clustercells_cov, seg_n, options)
    cluster_cell_imgtotal = cell_map(mask, clustercells_total, seg_n, options)
    cluster_cell_imguall = cell_map(mask, clustercells_uvall, seg_n, options)  # 0=use first segmentation to map
    clustercells_shape = cell_map(mask, clustercells_shapevectors, seg_n, options)
    # get markers for each respective cluster & then save the legend/markers
    print('Getting markers that separate clusters to make legend...')
    make_legends(im, filename, output_dir, options, clustercells_uvcenters, clustercells_covcenters, clustercells_totalcenters,
                 clustercells_uvallcenters, shapeclcenters)
    # save all clusterings to one csv
    print('Writing out all cell cluster IDs for all cell clusterings...')
    cell_cluster_IDs(filename, output_dir, options, clustercells_uv, clustercells_cov, clustercells_total, clustercells_uvall,
                     clustercells_shapevectors)
    # plots the cluster imgs for the best z plane
    print('Saving pngs of cluster plots by best focal plane...')
    plot_imgs(filename, output_dir, bestz, cluster_cell_imgu, cluster_cell_imgcov, cluster_cell_imguall,
              cluster_cell_imgtotal, clustercells_shape)
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
            (key, val) = line.split()
            val = is_number(val)
            options[key] = val
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

def showlegend(markernames: List[str], markertable: np.ndarray, outputfile: str, output_dir: Path ):
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
    plt.savefig(f, box_inches='tight')
    plt.close()


def find_locations(arr: np.ndarray) -> (np.ndarray, List[np.ndarray]):
    coords = np.indices(arr.shape).reshape(arr.ndim, arr.size)
    arr = arr.ravel()
    order = np.argsort(arr)
    arr = arr[order]
    coords = coords[:, order]
    locs = np.flatnonzero(np.diff(arr, prepend=arr[0] - 1))
    return arr[locs], np.split(coords, locs[1:], axis=1)

def cell_area_fraction(mask: MaskStruct) -> List[np.ndarray]:
    pass

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
    #img_files = img_files[0:3]
    
    snr_paths = output_dir / '*-SNR.csv'
    snr_files = get_paths(snr_paths)
      
    for i in range(0, len(snr_files)):
        df1 = pd.read_csv(snr_files[i])
        
        np1 = df1.to_numpy()
        zscore = np1[0,1:]
        otsu = np1[1,1:]
        
        if i == 0:
            f1 = zscore
            f2 = otsu
        else:
            f1 = np.vstack((f1, zscore))
            f2 = np.vstack((f2, otsu))
    
    if f1.ndim == 1 and f2.ndim == 1:
        df1 = pd.DataFrame([f1], columns = channel_n)
        df2 = pd.DataFrame([f2], columns = channel_n)
    else:
        df1 = pd.DataFrame(f1, columns = channel_n)
        df2 = pd.DataFrame(f2, columns = channel_n)
    
    df3 = pd.DataFrame({'Filename': img_files, 'Total Cells': total_cells})
    
    df1 = pd.concat([df3, df1], axis=1, sort=False)
    df2 = pd.concat([df3, df2], axis=1, sort=False)
    
    df1.to_csv(output_dir / 'summary_zscore.csv', index=False)
    df2.to_csv(output_dir / 'summary_otsu.csv', index=False)
