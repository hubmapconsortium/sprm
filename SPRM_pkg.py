from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
import os
import re
import glob
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import time
from sklearn.decomposition import PCA
import pandas as pd
from itertools import product, chain, combinations
from outlinePCA import *
import math

"""

Companion to SPRM.py
Package functions that are integral to running main script
Author:    Ted Zhang & Robert F. Murphy
01/21/2020 - 04/07/2020
Version: 0.50


"""


class IMGstruct:
    def __init__(self, path, options):
        self.img = self.read_img(path)
        self.data = self.read_data()
        self.path = path
        self.options = options
        self.name = self.set_name()
        self.channel_labels = self.read_channel_names()

    def set_data(self, data):
        self.data = data

    def set_img(self, img):
        self.img = img

    def get_data(self):
        return self.data

    def get_options(self):
        return self.options

    def get_meta(self):
        return self.img.metadata

    def quit(self):
        return self.img.close()

    def read_img(self, path):
        img = AICSImage(os.path.join(path))
        if not img.metadata:
            print('Metadata not found in input image')
            # might be a case-by-basis
            img = AICSImage(os.path.join(path), known_dims="CYX")
        else:
            print('Metadata found in input image')
        return img

    def read_data(self):
        data = self.img.data
        dims = data.shape
        s, t, c, z, y, x = dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
        if t > 1:
            data = data.reshape((s, 1, t * c, z, y, x))
        return data

    def read_channel_names(self):
        img = self.img
        return img.get_channel_names(scene=0)

    def set_name(self):
        s = self.path.split("/")
        return s[-1]

    def get_name(self):
        return self.name

    def get_channel_labels(self):
        return self.channel_labels


class MaskStruct:
    def __init__(self, path, options):
        self.img = self.read_img(path)
        self.data = self.read_data()
        self.path = path
        self.options = options
        self.labels = self.read_channel_names()
        self.bestz = self.get_bestz()

    def get_labels(self, label):
        return self.labels.index(label)

    def set_data(self, data):
        self.data = data

    def set_img(self, img):
        self.img = img

    def get_data(self):
        return self.data

    def get_options(self):
        return self.options

    def get_meta(self):
        return self.img.metadata

    def set_bestz(self, z):
        self.bestz = z

    def get_bestz(self):
        return self.bestz

    def quit(self):
        return self.img.close()

    def read_img(self, path):
        img = AICSImage(os.path.join(path))
        if not img.metadata:
            print('Metadata not found in mask image...')
            img = AICSImage(os.path.join(path), known_dims="CYX")
        print('Metadata found in mask image')
        return img

    def read_data(self):
        data = self.img.data
        dims = data.shape
        # s,t,c,z,y,x = dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]
        check = data[:, :, :, 0, :, :]
        check_sum = np.sum(check)
        if check_sum == 0:
            print('Duplicating best z to all z dimensions...')
            print(data.shape)
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
            print(bestz)
            # data now contains just the submatrix that has nonzero values
            # add back the z dimension
            data = x[:, :, :, np.newaxis, :, :]
            # print(data.shape)
            # and replicate it
            data = np.repeat(data, dims[3], axis=3)
            # print(data.shape)
        # set bestz
        self.set_bestz(bestz)
        return data

    def read_channel_names(self):
        img = self.img
        return img.get_channel_names(scene=0)


def calculations(coord, im, t):
    '''
        Returns covariance matrix, mean vector, and total vector
    '''
    
    z, y, x = coord[0], coord[1], coord[2]
    
    temp = im.get_data()
    
    channel_all_mask = temp[0,t,:,z,y,x]
    ROI = np.transpose(channel_all_mask)
    # stime = time.time()
    # dims = get_dims(ROI)
    # c, z, y, x = dims[0], dims[1], dims[2], dims[3]
    # #converted to 4D matrix
    # matrix_4D = ROI[:][:][:][:]
    # #reshape into a 2D array
    # matrix_c = matrix_4D.reshape((c,z*y*x))
    # mean vector calculation
    # mu_v = np.reshape(np.mean(ROI,axis=1),(matrix_c.shape[0],1))
    # mu_v = np.mean(matrix_c,axis=1)
    # covariance matrix calculation
    cov_m = np.cov(ROI)
    mu_v = np.reshape(np.mean(ROI, axis=1), (ROI.shape[0], 1))
    total = np.reshape(np.sum(ROI, axis=1), (ROI.shape[0], 1))
    #    if ROI_idx == 1:
    #        print(ROI.shape)
    #        print('COVARIANCE MATRIX: ROI-'+str(ROI_idx)+' time dim:'+str(t+1))
    #        print(cov_m.shape)
    #        print(cov_m)
    #        print('Mean vector:')
    #        print(mu_v.shape)
    #        print(mu_v)
    if not cov_m.shape:
        cov_m = np.array([cov_m])
        print(cov_m)
        print(cov_m.shape)
    # cov_mu = np.concatenate((cov_m,mu_v), axis=1)
    # print(cov_mu)
    # print(cov_mu.shape)
    # breakpoint()
    # covar_path = './covariance_matrices'
    # hard coded check
    # if not os.path.exists(covar_path):
    #    os.mkdir(covar_path)
    # split img_file to get img name
    # img_name = img_file.split('/')
    # write_2_file(cov_m, covar_path+'/'+img_name[-1]+'__ROI-'+str(ROI_idx)+'_tdim_'+str(t))
    # print('Elapsed time for mean, total and covar calculations: ', time.time() - stime)
    return cov_m, mu_v, total


def cell_cluster(cell_matrix, segnum, options):
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
        print('segmentation channel: ' + str(segnum + 1))
        print(cell_matrix.shape)
        cell_matrix = cell_matrix[segnum, :, :, :]
        print(cell_matrix.shape)
        print('3D cell matrix')
        cell_matrix = cell_matrix.reshape((cell_matrix.shape[0], cell_matrix.shape[1] * cell_matrix.shape[2]))
    else:
        cshape = cell_matrix.shape
        temp_matrix = np.zeros([cshape[1], cshape[2], cshape[3], cshape[0]])
        for i in range(0, cell_matrix.shape[0]):
            temp_matrix[:, :, :, i] = cell_matrix[i, :, :, :]
        cell_matrix = temp_matrix
        cell_matrix = cell_matrix.reshape(
            (cell_matrix.shape[0], cell_matrix.shape[1] * cell_matrix.shape[2] * cell_matrix.shape[3]))
        # dims = get_dims(cell_matrix)
        # cl, cha, chb  = dims[0], dims[1], dims[2]
        # cell_matrix_u = np.reshape(cell_matrix, (cl,t*cha))
    print(cell_matrix.shape)
    # kmeans clustering
    print('Clustering cells...')
    cellbycluster = KMeans(n_clusters=options.get("num_cellclusters"), random_state=0).fit(cell_matrix)
    # returns a vector of len cells and the vals are the cluster numbers
    cellbycluster_labels = cellbycluster.labels_
    print(cellbycluster_labels.shape)
    # print(cellbycluster_labels)
    # print(len(np.unique(cellbycluster_labels)))
    clustercenters = cellbycluster.cluster_centers_
    print(clustercenters.shape)
    return cellbycluster_labels, clustercenters


# def map: cell index to cluster index. Mask indexed img changes that to cluster number (mask,cellbycluster)
def cell_map(mask, cc_v, seg_n):
    '''
    Maps the cells to indexed img
    '''
    print('Mapping...')
    mask_img = mask.get_data()
    mask_img = mask_img[0, 0, seg_n, :, :, :]
    temp = mask_img.copy()
    # cluster_img = np.zeros(mask_img.shape)
    # start_time = time.time()
    cc_v += 1
    clusters = np.unique(cc_v)
    
    
    # for i in range(0, len(cc_v)):
    #     coord = np.where(mask_img == i + 1)
    #     cluster_img[coord[0], coord[1], coord[2]] = cc_v[i] + 1
    # # cluster_imgt = [mask_img == i+1]
    # # cluster_imgt = cluster_img * (cc_v[i]+1)
    # elapsed_time2 = time.time() - start_time
    
    stime = time.time()
    for i in range(0, len(clusters)):
        cell_num = np.where(cc_v == clusters[i])[0] + 1
        bit_mask = np.isin(mask_img, cell_num)
        temp[bit_mask] = clusters[i]
    
    print('Elapsed time for cell mapping <vectorized>: ', time.time() - stime)
    # print('Elapsed time for cell mapping <loop>: ', time.time() - start_time)
    return temp
    # return cluster_img


def mask_img(mask, dir, j, options):
    '''
    returns: a 3D matrix that represents the jth segmented image
    '''
    # mask_path = os.path.join(dir+'/'+mask_file)
    # mask = IMGstruct(mask_file, options)
    sMask = mask.get_data()
    if sMask.shape[3] > 1:
        # print(sMask.shape)
        sMask = sMask[0, 0, j, :, :, :]
        unique = np.unique(sMask)
        # look through z slices for best z slice and get it
        # for i in range(0,mask.shape[0]):
        #     x= mask[i][:][:]
        #     y= np.sum(x)
        #     print(x)
        #     print(y)
        #     if y > 0: 
        #         z = i
        #         break
        #     else:
        #         continue
        # 3D matrix now - 3rd dim is channels
        #        mask_3D = mask_4D[:,z,:,:]
        #        tmask_3D = mask_3D
        #        #print(tmask_3D[2])
        # filter/thresholds/fill_mask
        #        for j in range(0,mask_3D.shape[0]):
        #            tmask_3D[j] = cv2.cvtColor(tmask_3D[j], cv2.COLOR_BGR2GRAY)
        #            tmask_3D[j] = cv2.GaussianBlur(mask_3D[j],(5,5),3)
        #            ret , tmask_3D[j] = cv2.threshold(tmask_3D[j], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #            tmask_3D[j] = fill_holes(tmask_3D[j])
        #            plt.imshow(tmask_3D[j], cmap='gray', interpolation='nearest')
        #            plt.show()
        #            breakpoint()
        # duplicate the best z for all z
        # mask_3D = mask[z,:,:]
        # getting all the ROIs - indexes 0 - n of all objs
        # mask_3D = mask_3D[np.newaxis,:,:]
        # mask_3D = np.repeat(mask_3D,12, axis=0)
        # print(mask_3D.shape)
        return sMask, unique
    else:
        print('Mask is 2D, SPRM requires a 3D mask')
        exit()


def get_masked_imgs(labeled_mask, maskIDs):
    '''
        Returns the masked image as a 4D image ---> this might change as it creates more memory in allocation
    '''
    # img = im.get_data()
    masked_imgs = []
    for i in range(1, len(maskIDs)):

        coor = np.where(labeled_mask == maskIDs[i])
        # z, y, x = coor[0], coor[1], coor[2]

        # temp = im.get_data()
        
        # channel_all_mask = temp[0, t, :, z, y, x]
        # channel_all_mask = np.transpose(channel_all_mask)

        masked_imgs.append(coor)

    return masked_imgs


def SRM(img_files, mask_files, options):
    '''
        Matlab wrapper for python
    '''
    eng = matlab.engine.start_matlab("-desktop")
    eng.cd('./SRMcode')
    answer = eng.main_HPA(img_files, mask_files, options, nargout=1)
    # print(answer)


def sort(l):
    '''
        Sorts the 
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_imgs(img_dir):
    
    try:
        img_files = sort(os.listdir(img_dir))
        # hidden file in mac_os system will raise unexpected file error
        if ".DS_Store" in img_files:
            img_files.remove(".DS_Store")
        img_files = [img_dir + '/' + i for i in img_files]
        img_files = [os.path.abspath(i) for i in img_files]
        
    except FileNotFoundError:
        s = img_dir.split('/')
        img_dir = '/'.join(s[:-1])
        pattern = s[-1]
        img_files = glob.glob(img_dir+'/'+pattern)
        img_files = sort(img_files)
    
    return img_files

def get_regex_files(pattern, directory):
    
    pass

def get_df_format(sub_matrix, s, img, options):
    # can use dict switch to speed up access if the formats get > 5
    names = img.get_channel_labels()
    if len(sub_matrix.shape) < 3:
        # if 'mean' in s:
        #     header_num = np.sum(sub_matrix, axis = 0) / sub_matrix.shape[0]
        # elif 'total' in s:
        #     header_num = np.sum(sub_matrix, axis = 0) 
        # header_num = np.array(["%f" % w for w in header_num.reshape(header_num.shape)])
        # header_num = header_num.reshape(header_num.shape)
        # header_list = header_num.tolist()
        # header = [i + j for i, j in zip([s + ': ' for s in names] ,header_list)]
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
    return header, sub_matrix, s


def write_2_file(sub_matrix, s, img, options):
    header, sub_matrix, s = get_df_format(sub_matrix, s, img, options)
    write_2_csv(header, sub_matrix, s)


def write_2_csv(header, sub_matrix, s):
    df = pd.DataFrame(sub_matrix, index=list(range(1, sub_matrix.shape[0] + 1)))
    print(df)
    df.to_csv(s + '.csv', header=header)
    # with open(s+'.csv', mode='w') as dm:
    #     writer = csv.writer(dm, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    #     if len(list) == 1:
    #         writer.writerow(list)
    #     else:
    #         writer.writerow(img.get_channel_labels())
    #         for i in range(0, len(list)):
    #             writer.writerow(list[i])


def build_matrix(im, mask, masked_imgs_coord, j, omatrix):
    if j == 0:
        return np.zeros((im.data.shape[1], mask.data.shape[2], len(masked_imgs_coord), im.data.shape[2], im.data.shape[2]))
    else:
        return omatrix


def build_vector(im, mask, masked_imgs_coord, j, omatrix):
    if j == 0:
        return np.zeros((im.data.shape[1], mask.data.shape[2], len(masked_imgs_coord), im.data.shape[2], 1))
    else:
        return omatrix


def clusterchannels(im, options):
    '''
        cluster all channels using PCA
    '''
    print('In clusterchannels')
    print(im.data.shape)
    pca_channels = PCA(n_components=options.get("n_components"), svd_solver='full')
    channvals = im.data[0, 0, :, :, :, :]
    print(channvals.shape)
    keepshape = channvals.shape
    channvals = channvals.reshape(channvals.shape[0], channvals.shape[1] * channvals.shape[2] * channvals.shape[3])
    print(channvals.shape)
    channvals = channvals.transpose()
    print(channvals.shape)
    reducedim = pca_channels.fit_transform(channvals)
    print(pca_channels)
    print(pca_channels.explained_variance_ratio_)
    print(pca_channels.singular_values_)
    print(reducedim.shape)
    reducedim = reducedim.transpose()
    print(reducedim.shape)
    reducedim = reducedim.reshape(reducedim.shape[0], keepshape[1], keepshape[2], keepshape[3])
    print(reducedim.shape)
    return reducedim


def plotprincomp(reducedim, bestz, filename):
    print('In plotprincomp')
    print(reducedim.shape)
    reducedim = reducedim[:, bestz, :, :]
    k = reducedim.shape[0]
    if k > 2:
        reducedim = reducedim[0:3, :, :]
    else:
        rzeros = np.zeros(reducedim.shape)
        reducedim[k:3, :, :] = rzeros[k:3, :, :]
    print(reducedim.shape)
    plotim = reducedim.transpose(1, 2, 0)
    print(plotim.shape)
    for i in range(0, 3):
        cmin = plotim[:, :, i].min()
        cmax = plotim[:, :, i].max()
        print(cmin, cmax)
        plotim[:, :, i] = ((plotim[:, :, i] - cmin) / (cmax - cmin)) * 255.
        cmin = plotim[:, :, i].min()
        cmax = plotim[:, :, i].max()
        print(cmin, cmax)
    plotim = plotim.round()
    plotim = plotim.astype(int)
    plt.imshow(plotim)
    plt.show(block=False)
    plt.savefig(filename, box_inches='tight')
    return plotim


def voxel_cluster(im, options):
    '''
    cluster multichannel image into superpixels
    '''
    print('In voxel_cluster')
    print(im.data.shape)
    if im.data.shape[0] > 1:
        print('image has more than one time point')
        print('Have not implemented support yet..')
        exit()
    channvals = im.data[0, 0, :, :, :, :]
    print(channvals.shape)
    keepshape = channvals.shape
    channvals = channvals.reshape(channvals.shape[0], channvals.shape[1] * channvals.shape[2] * channvals.shape[3])
    print(channvals.shape)
    channvals = channvals.transpose()
    print(channvals.shape)
    # get random sampling of pixels in 2d array
    np.random.seed(0)
    sampling = float(options.get('precluster_sampling'))
    samples = math.ceil(sampling * channvals.shape[0])
    # lower bound threshold on pixels
    if samples < options.get("precluster_threshold"):
        samples = options.get("precluster_threshold")
    idx = np.random.choice(channvals.shape[0], samples)
    channvals_random = channvals[idx]
    print(channvals_random.shape)
    # kmeans clustering of random sampling
    print('Clustering random sample of voxels...')
    stime = time.time()
    voxelbycluster = KMeans(n_clusters=options.get("num_voxelclusters"), random_state=0).fit(channvals_random)
    print('random sample voxel cluster runtime: ' + str(time.time() - stime))
    cluster_centers = voxelbycluster.cluster_centers_
    # fast kmeans clustering with inital centers
    print('Clustering voxels with initialized centers...')
    stime = time.time()
    voxelbycluster = KMeans(n_clusters=options.get("num_voxelclusters"), init=cluster_centers, random_state=0, max_iter=100, verbose=0, n_init=1).fit(channvals)
    #voxelbycluster = KMeans(n_clusters=options.get("num_voxelclusters"), random_state=0).fit(channvals)
    print('Voxel cluster runtime: ', time.time() - stime)
    # returns a vector of len number of voxels and the vals are the cluster numbers
    voxelbycluster_labels = voxelbycluster.labels_
    print(voxelbycluster_labels.shape)
    voxelbycluster_labels = voxelbycluster_labels.reshape(keepshape[1], keepshape[2], keepshape[3])
    print(voxelbycluster_labels.shape)
    print('Number of unique labels:')
    print(len(np.unique(voxelbycluster_labels)))
    return voxelbycluster_labels


def findmarkers(clustercenters, options):
    '''
        find a set of markers that have the largest variance without correlations among them
    '''
    print(clustercenters.shape)
    clustercenters = np.transpose(clustercenters)
    print(clustercenters.shape)
    covar = np.cov(clustercenters)
    print(covar.shape)
    print(covar)
    varianc = np.diagonal(covar).copy()
    print(varianc.shape)
    print(varianc)
    cc = np.corrcoef(clustercenters)
    print(cc.shape)
    print(cc)
    thresh = 0.9
    increment = 0.1
    lowerthresh = 0.5
    markergoal = options.get('num_markers')
    vartemp = varianc.copy()
    cctemp = cc.copy()
    markerlist = []
    while True:
        hivar = np.argmax(vartemp)
        # print(hivar)
        if vartemp[hivar] == 0:
            if len(markerlist) == markergoal:
                print('Just right')
                print(markerlist)
                return markerlist
            else:
                if len(markerlist) > markergoal:
                    print('Too many')
                    thresh = thresh - increment
                    if increment < 0.001:
                        print('Truncating')
                        markerlist = markerlist[0:markergoal]
                        print(markerlist)
                        return markerlist
                else:
                    if len(markerlist) < markergoal:
                        print('Not enough')
                        increment = increment / 2
                        thresh = thresh + increment
                # reset to redo search
                print('Resetting')
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
            print('Continuing')
            print(markerlist)


def matchNShow_markers(clustercenters, markerlist, features):
    '''
        get the markers to indicate what the respective clusters represent
    '''
    # markers = img.get_channel_labels()
    # try:
    #     markers = [markers[i] for i in markerlist]
    # except IndexError:
    #     markers = options.get('channel_label_combo')
    #     markers = [markers[i] for i in markerlist]

    markers = [features[i] for i in markerlist]
    print(clustercenters.shape)
    table = clustercenters[:, markerlist]
    print(markerlist)
    print(table)
    return table, markers


def write_ometiff(im, *argv):
    # img = im.get_data()
    # print(img.shape)
    print(len(argv))
    pcaimg = argv[0]
    print(pcaimg.shape)
    pcaimg = pcaimg.reshape((pcaimg.shape[2], pcaimg.shape[0], pcaimg.shape[1]))
    print(pcaimg.shape)
    pcaimg = pcaimg.astype(np.int32)
    superpixel = argv[1].astype(np.int32)
    superpixel = superpixel[np.newaxis, :, :]
    # print(type(superpixel))
    # print(superpixel.dtype.name)
    print(superpixel.shape)
    s = ["-channel_pca.ome.tiff", "-superpixel.ome.tiff"]
    # argv[0] = pcaimg
    #    print(argv[1].shape)
    #    print(argv[2].shape)
    #     visualByChannel = np.zeros((len(argv),pcaimg.shape[0],img.shape[4],img.shape[5]))
    #     #print(visualByChannel)
    # #   visualByChannel = visualByChannel[0,:,len(argv),pcaimg.shape[2],:,:]
    #     visualByChannel[0,:,:,:] = pcaimg
    #     #channel loop
    #     for i in range(1,visualByChannel.shape[0]):
    #         visualByChannel[i,:,:,:] = argv[i]
    #     print(visualByChannel.shape)
    check_file_exist(im, s)
    writer = OmeTiffWriter(im.get_name() + s[0])
    writer.save(pcaimg, channel_names=im.get_channel_labels(), image_name=im.get_name())
    writer = OmeTiffWriter(im.get_name() + s[1])
    writer.save(superpixel, channel_names=im.get_channel_labels(), image_name=im.get_name())


def check_file_exist(im, s):
    for i in range(0, len(s)):
        if os.path.isfile(im.get_name() + s[i]):
            os.remove(im.get_name() + s[i])
        else:
            continue


# def save_files(*argv):
#     data_path = './visualization_data'
#     #hard coded check
#     if not os.path.exists(data_path):
#         os.mkdir(data_path)
#         os.chdir(data_path)
#     else:
#         os.chdir(data_path)
#     np.savez(data_path, *argv)
#     os.chdir('..')

def cell_cluster_IDs(filename, *argv):
    allClusters = argv[0]
    for idx in range(1, len(argv)):
        allClusters = np.column_stack((allClusters, argv[idx]))
    # hard coded --> find a way to automate the naming
    write_2_csv(list(['Mean', 'Covariance', 'Total', 'Mean All', 'Shape Vectors']), allClusters,
                filename + '-cell_cluster')
    # write_2_csv(list(['cluster ID']),clustercells_uv,filename+'-clustermeansjustbycellmask')
    # write_2_csv(list(['cluster ID']),clustercells_cov,filename+'-clustercovarjustbycellmask')
    # write_2_csv(list(['cluster ID']),clustercells_total,filename+'-clustertotalsbycellmask')
    # write_2_csv(list(['cluster ID']),clustercells_uvall,filename+'-clustermeansbyallmasks')
    # write_2_csv(list(['cluster ID']),clustercells_shapevectors,filename+'-clusterbyshape')

def plot_img(cluster_im, filename):
    plt.imshow(cluster_im, interpolation='nearest')
    plt.show(block=False)
    plt.savefig(filename, box_inches='tight')
    
def plot_imgs(filename, *argv):
    plot_img(argv[0], filename + '-ClusterByMeansPerCell.png')
    plot_img(argv[1], filename + '-ClusterByCovarPerCell.png')
    plot_img(argv[2], filename + '-ClusterByMeansAllMasks.png')
    plot_img(argv[3], filename + '-ClusterByTotalPerCell.png')
    plot_img(argv[4], filename + '-ClusterByShape.png')


def make_legends(im, filename, options, *argv):
    feature_names = im.get_channel_labels()
    feature_covar = options.get('channel_label_combo')
    feature_meanall = feature_names + feature_names + feature_names + feature_names
    feature_shape = ['shapefeat ' + str(ff) for ff in range(0, argv[4].shape[1])]

    print('Finding cell mean cluster markers...')
    retmarkers = findmarkers(argv[0], options)
    table, markers = matchNShow_markers(argv[0], retmarkers, feature_names)
    write_2_csv(markers, table, filename + '-clustercells_cellmean_legend')

    print('Finding cell covariance cluster markers...')
    retmarkers = findmarkers(argv[1], options)
    table, markers = matchNShow_markers(argv[1], retmarkers, feature_covar)
    write_2_csv(markers, table, filename + '-clustercells_cellcovariance_legend')

    print('Finding cell total cluster markers...')
    retmarkers = findmarkers(argv[2], options)
    table, markers = matchNShow_markers(argv[2], retmarkers, feature_names)
    write_2_csv(markers, table, filename + '-clustercells_celltotal_legend')

    print('Finding cell mean ALL cluster markers...')
    retmarkers = findmarkers(argv[3], options)
    table, markers = matchNShow_markers(argv[3], retmarkers, feature_meanall)
    write_2_csv(markers, table, filename + '-clustercells_cellmeanALL_legend')

    print('Finding cell shape cluster markers...')
    retmarkers = findmarkers(argv[4], options)
    table, markers = matchNShow_markers(argv[4], retmarkers, feature_shape)
    write_2_csv(markers, table, filename + '-clustercells_cellshape_legend')


def save_all(filename, im, seg_n, options, *argv):
    # hard coded for now
    mean_vector = argv[0]
    covar_matrix = argv[1]
    total_vector = argv[2]
    outline_vectors = argv[3]
    write_2_file(mean_vector[0, seg_n, :, :, 0], filename + '-cell_channel_mean', im, options)
    write_2_file(covar_matrix[0, seg_n, :, :, :], filename + '-cell_channel_covar', im, options)
    write_2_file(total_vector[0, seg_n, :, :, 0], filename + '-cell_channel_total', im, options)
    write_2_file(outline_vectors, filename + '-cell_shape', im, options)


def cell_analysis(im, mask, filename, bestz, seg_n, options, *argv):
    '''
        cluster and statisical analysis done on cell:
        clusters/maps and makes a legend out of the most promient channels and writes them to a csv
    '''
    stime = time.time()
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
    clustercells_uvall, clustercells_uvallcenters = cell_cluster(mean_vector, -1, options)      # -1 means use all segmentations
    clustercells_shapevectors, shapeclcenters = shape_cluster(shape_vectors, options)
    # map back to the mask segmentation of indexed cell region
    print('Mapping cell index in segmented mask to cluster IDs...')
    cluster_cell_imgu = cell_map(mask, clustercells_uv, seg_n)
    cluster_cell_imgcov = cell_map(mask, clustercells_cov, seg_n)
    cluster_cell_imgtotal = cell_map(mask, clustercells_total, seg_n)
    cluster_cell_imguall = cell_map(mask, clustercells_uvall, seg_n)  # 0=use first segmentation to map
    clustercells_shape = cell_map(mask, clustercells_shapevectors, seg_n)
    # get markers for each respective cluster & then save the legend/markers
    print('Getting markers for separate cluster to make legend...')
    make_legends(im, filename, options, clustercells_uvcenters, clustercells_covcenters, clustercells_totalcenters,
                 clustercells_uvallcenters, shapeclcenters)
    # save each cluster to one csv
    print('Writing out all cell cluster IDs for each different cell clusters...')
    cell_cluster_IDs(filename, clustercells_uv, clustercells_cov, clustercells_total, clustercells_uvall,
                     clustercells_shapevectors)
    # plots the cluster imgs for the best z plane
    print('Saving pngs of cluster plots by best focal plane...')
    plot_imgs(filename, cluster_cell_imgu[bestz], cluster_cell_imgcov[bestz], cluster_cell_imguall[bestz],
              cluster_cell_imgtotal[bestz], clustercells_shape[bestz])
    print('Elapsed time for cluster img saving: ', time.time() - stime)


def make_DOT(mc, fc, coeffs, ll):
    pass


def powerset(iterable):
    '''
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def read_options(options_path):
    #read in options
    options = {}
    with open(options_path) as f:
        for line in f:
            (key, val) = line.split()
            val = is_number(val)
            options[key] = val
    return options

def is_number(val):
    try:
        val = int(val)
    except ValueError:
        try: 
            val = float(val)
        except ValueError:
            val = val
    return val

def check_results_dir():
    path = './results'
    if os.path.exists(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)
    return os.getcwd()