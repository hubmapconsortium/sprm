from datetime import datetime
from math import floor
from typing import List

import numba as nb
import numpy as np
from scipy.sparse import coo_matrix, find, lil_matrix
from sklearn import preprocessing

"""

Package for reallocating intensity from IMS image of lower resolution
than a corresponding cell mask image
Author: Robert F. Murphy and Ted (Ce) Zhang
5/21/2020 - 12/23/2025
Version: 2.0.1


"""


def downsampleIMS(im, factor):
    imshape = im.shape
    # print(imshape)
    imshapenew = [int(ii / factor) for ii in imshape]
    imnew = np.zeros(imshapenew)
    for i in range(0, imshapenew[0]):
        for j in range(0, imshapenew[1]):
            imnew[i, j] = im[i * factor, j * factor]
    # print(imnew.shape)
    return imnew


# @nb.jit(nopython=True, parallel=True, fastmath=True)
def maxIMSsize(ROI: int, rshape: int, mshape: int) -> int:
    col = floor((ROI / rshape) * (mshape / rshape))
    row = floor((ROI % rshape) * (mshape / rshape))
    j = col * mshape + row
    return j


def findpixelfractions(ROI: np.ndarray, rshape: tuple, mshape: tuple, c: int):
    # print(ROI.shape)
    # print('maxROI = ',max(ROI))
    reducedsize = maxIMSsize(ROI.shape[0], rshape[0], mshape[0])
    # print(reducedsize)
    # Y is the number of small pixels in a given cell from a given large pixel
    # Y = np.zeros([max(ROI) + 1, reducedsize])
    # Y = lil_matrix((max(ROI)+1, reducedsize), dtype='float32')
    # Y = coo_matrix((max(ROI) + 1, reducedsize), dtype='float32')
    # Y1 = coo_matrix((max(ROI) + 1, reducedsize), dtype='float32')
    # Y = Y.tolil()
    # Y1 = Y1.tolil()
    # X is fraction of area of large pixel to allocate to a given cell
    # X = np.zeros(Y.shape)
    # X = lil_matrix(Y.shape, dtype='float32')
    # X = coo_matrix((max(ROI) + 1, reducedsize)), dtype='float32')
    # X = X.tolil()
    # A is fraction of area of a cell from a given pixel
    # A = np.zeros(Y.shape)
    # A = lil_matrix(Y.shape, dtype='float32')
    A = coo_matrix((max(ROI) + 1, reducedsize), dtype="float32")
    A = A.tolil()

    col1 = np.arange(ROI.shape[0])
    col1 = IMSmap_new(col1, rshape[0], mshape[0])
    # Don't need unless mask and mxif are different shapes
    # col1 = np.where(col1 >= reducedsize, reducedsize - 1, col1)
    data = np.ones(ROI.shape[0])
    Y = coo_matrix((data, (ROI, col1)), shape=(max(ROI) + 1, reducedsize)).tolil()

    # Y1[ROI, col1] += 1

    # col2 = []
    # for k in range(0, ROI.shape[0]):
    #     col = IMSmap(k, rshape[0], mshape[0])
    #     # threshold
    #     if col >= reducedsize:
    #         col = reducedsize - 1
    #
    #     col2.append(col)
    #     Y[ROI[k], col] = Y[ROI[k], col] + 1
    # print(Y)
    denomX = np.prod(rshape) / np.prod(mshape)
    # print(denomX)
    # sumY1 should be equal to denomX - the sum of contributions from each pixel
    # across all cells including "cell 0=background" should be constant
    # sumY1 = np.sum(Y, axis=0)  # should be equal to denomX
    # (the contributions across all cells including "cell 0"
    # print('sumY1=',sumY1)
    # sumY2 should be the area of each cell
    sumY2 = np.asarray(np.sum(Y, axis=1)).reshape(-1)
    # sumY2 = np.asarray(sumY2).reshape(-1)
    # print('sumY2=',sumY2)

    # possibly need to change to lil_matrix to csr?

    # denomA = Y.sum(axis=1)
    X = Y / denomX
    X_CH = [X] * c

    indices = np.where(sumY2 > 0)
    # all cells have an area and are not empty
    # if len(indices[0]) == len(sumY2):
    #     A = Y.multiply(1/sumY2)  # changes to coo_matrix
    #     A = A.tolil()
    # else:
    #     sumY2 = np.asarray(sumY2).reshape(-1)
    #     A[indices[0], :] = Y[indices[0], :] / sumY2
    #     #A = ploop_nonzero_cells(A, Y, sumY2, indices)

    # sumY2 = np.asarray(sumY2).reshape(-1)
    Y_nnz = Y.nonzero()
    coords = np.nonzero(np.isin(Y_nnz[0], indices[0]))[0]

    # if Y_nnz[0].shape[0] != coords.shape[0]:
    #     pass
    # else:
    Y_nnz_trimi = Y_nnz[0][coords]
    Y_nnz_trimj = Y_nnz[1][coords]
    divsumY2 = sumY2[Y_nnz_trimi]
    dataY = Y[Y_nnz_trimi, Y_nnz_trimj].toarray().reshape(-1)
    dataY = dataY / divsumY2
    A = coo_matrix((dataY, (Y_nnz_trimi, Y_nnz_trimj)), shape=(max(ROI) + 1, reducedsize)).tolil()
    # A1 = Y[indices[0], :].multiply(1 / sumY2)
    # A1 = np.where(sumY2 > 0, Y / sumY2, A)

    # for i in range(0, max(ROI) + 1):
    #     denomA = sum([Y[i, k] for k in range(0, reducedsize)])
    #     # print(denomA)
    #     for j in range(0, reducedsize):
    #         X[i, j] = Y[i, j] / denomX
    #         if denomA > 0:
    #             A[i, j] = Y[i, j] / denomA
    # sumX = np.sum(X, axis=0)  # sum of fractions for each pixel - should be 1's
    # print('sumX=',sumX)
    # sumA = np.sum(A, axis=1)  # sum of fractions for each cell - should be 1's
    # print('sumA=',sumA)
    # sumY2 = np.asarray(sumY2).reshape(-1)

    return X_CH, A, sumY2, reducedsize


# @nb.njit(parallel=True, fastmath=True)
# def ploop_nonzero_cells(A, Y, sumY2, indices):
#     for i in nb.prange(len(indices[0])):
#         A[i, :] = Y[i, :] / sumY2
#
#     return A


@nb.njit(parallel=True, fastmath=True)
def IMSmap_new(k: np.ndarray, rshape: int, mshape: int) -> np.ndarray:
    col = np.floor((k / rshape) * (mshape / rshape))
    row = np.floor((k % rshape) * (mshape / rshape))
    j = (col * mshape) + row
    return j.astype(np.int64)


# def IMSmap(k: int, rshape: int, mshape: int) -> int:
#     col = floor((k / rshape) * (mshape / rshape))
#     row = floor((k % rshape) * (mshape / rshape))
#     j = (col * mshape) + row
#     return j


def calcM(
    ROI: np.ndarray,
    ROIshape,
    M,
    Mshape,
    X,
    meanIntenCell,
    averagebackground,
    smallpixelsperbigpixel,
    chidx,
):
    # newM = [0] * ROI.shape[0]
    newM = np.zeros((M.shape[0], ROI.shape[0]), dtype=np.uint32)
    # newM1 = np.zeros(ROI.shape, dtype=np.uint32)
    # smallpixelsperbigpixel = np.prod(ROIshape) / reducedsize

    # for k in range(0, ROI.shape[0]):
    #     curcell = ROI[k]
    #     if curcell == 0 and not averagebackground:
    #         j = IMSmap(k, ROIshape[0], Mshape[0])
    #         newM[k] = X[curcell, j] * M[j] / smallpixelsperbigpixel
    #     else:
    #         newM[k] = meanIntenCell[curcell]
    # ROI = np.where(ROI >= X.shape[0], X.shape[0] - 1, ROI)
    indices = np.arange(ROI.shape[0])
    newM[:, indices] = meanIntenCell[:, ROI]

    # print(X.shape)
    # print(newM.shape)
    # print(M.shape)
    # print(reducedsize)
    # print(max(ROI))
    if averagebackground:
        bgindices = np.where(ROI == 0)
        j = IMSmap_new(bgindices[0], ROIshape[0], Mshape[1])
        # j = np.where(j >= reducedsize, reducedsize - 1, j)
        # loop - might be a more efficient option
        for i in range(0, M.shape[0]):
            newM[i, bgindices[0]] = X[i][0, j] * M[i, j] / smallpixelsperbigpixel

    return newM


@nb.njit(parallel=True, fastmath=True)
def calc_totalIntensity(totInten, XM, ROI):
    for j in nb.prange(XM.shape[0]):
        for i in nb.prange(ROI.shape[0]):
            totInten[j, ROI[i]] += XM[j, i]

    return totInten


def updateXM(X, M, A, relevantpixels):
    # get only relevantpixels product of X * M
    Xreal = X[0][relevantpixels[0], relevantpixels[1]]
    Mreal = M[:, relevantpixels[1]]
    Areal = A[relevantpixels[0], relevantpixels[1]].toarray().reshape(-1)
    Xreal = Xreal.toarray().reshape(-1)
    # XM = np.zeros((M.shape[0], Areal.shape[0]))
    # channel_multi(Xreal, Mreal, XM)
    # XM = Xreal.multiply(Mreal)
    XM = np.multiply(Mreal, Xreal[np.newaxis, :])

    # find 0s
    idx = set(np.arange(len(relevantpixels[0])))
    XM_zero = np.where(XM != 0)
    XM_zero = set(XM_zero[1])
    idx = list(XM_zero.intersection(idx))

    # update all features
    relevantpixels[1] = relevantpixels[1][idx]
    relevantpixels[0] = relevantpixels[0][idx]
    XM = XM[:, idx]
    Xreal = Xreal[idx]
    Mreal = Mreal[:, idx]
    Areal = Areal[idx]

    return XM, Areal, Mreal


# @nb.njit(parallel=True)
# def fill_channel_idx(channel_idx):
#     for i in nb.prange(channel_idx.shape[0]):
#         channel_idx[i, :] = i
def estimatedeltas(
    relevantpixels: List, maxROI: int, reducedsize: int, XM, M, Areal, Mreal, chidx
):
    # totInten = [0] * (maxROI + 1)
    totInten = np.zeros((M.shape[0], maxROI + 1))
    # meanIntenPixel = lil_matrix((maxROI + 1, reducedsize))
    # deltax1 = lil_matrix((maxROI + 1, reducedsize))

    # get only relevantpixels product of X * M
    # Xreal = X[relevantpixels[0], relevantpixels[1]]
    # Mreal = M[:, relevantpixels[1]]
    # Areal = A[relevantpixels[0], relevantpixels[1]].toarray().reshape(-1)
    # Xreal = Xreal.toarray().reshape(-1)
    # # XM = np.zeros((M.shape[0], Areal.shape[0]))
    # # channel_multi(Xreal, Mreal, XM)
    # # XM = Xreal.multiply(Mreal)
    # XM = np.multiply(Mreal, Xreal[np.newaxis, :])
    #
    # #find 0s
    # idx = set(np.arange(len(relevantpixels[0])))
    # XM_zero = np.where(XM != 0)
    # XM_zero = set(XM_zero[1])
    # idx = list(XM_zero.intersection(idx))
    #
    # #update all features
    # relevantpixels[1] = relevantpixels[1][idx]
    # relevantpixels[0] = relevantpixels[0][idx]
    # XM = XM[:, idx]
    # Xreal = Xreal[idx]
    # Mreal = Mreal[:, idx]
    # Areal = Areal[idx]

    XMdA = XM / Areal  # meanIntenPixel
    # XMarr = XM.toarray().reshape(-1)
    # s = time.monotonic()
    calc_totalIntensity(totInten, XM, relevantpixels[0])

    # calc deltax
    temp = totInten[:, relevantpixels[0]] - XMdA
    deltaxReal = temp * Areal
    deltaxReal = np.divide(deltaxReal, Mreal, out=np.zeros_like(Mreal), where=Mreal != 0)
    # f = time.monotonic() - s

    # make array of channel idx
    # channel_idx = np.zeros((XM.shape))
    # fill_channel_idx(channel_idx)
    # #duplicate relevantpixels
    # xrep = np.repeat(relevantpixels[0][np.newaxis, :], XM.shape[0], axis=0)
    # yrep = np.repeat(relevantpixels[1][np.newaxis, :], XM.shape[0], axis=0)
    #
    # #flatten/vectorize
    # channel_idx = channel_idx.reshape(-1)
    # xrep = xrep.reshape(-1)
    # yrep = yrep.reshape(-1)
    #
    # XMdA = XMdA.reshape(-1)
    # deltaxReal = deltaxReal.reshape(-1)

    # construct sparse matrix from the data
    meanIntenPixel = []
    deltax = []
    for i in range(M.shape[0]):
        a = coo_matrix(
            (XMdA[i, :], (relevantpixels[0], relevantpixels[1])), shape=(maxROI + 1, reducedsize)
        )
        b = coo_matrix(
            (deltaxReal[i, :], (relevantpixels[0], relevantpixels[1])),
            shape=(maxROI + 1, reducedsize),
        )

        meanIntenPixel.append(a.tolil())
        deltax.append(b.tolil())

    ###################################################################################
    # totInten = np.zeros(maxROI + 1)
    # deltax = lil_matrix((maxROI + 1, reducedsize))
    # s1 = time.monotonic()
    # for i in nonemptycellindices:
    #     # for j in nonemptypixels:
    #     for j in relevantpixels[i]:
    #         #if A[i, j] > 0:
    #         totInten[i] = totInten[i] + (X[i, j] * M[j])
    #         meanIntenPixel[i, j] = X[i, j] * M[j] / A[i, j]
    #         # calculate how much the fractions should change
    #         # in order for each pixel's contribution to get
    #         # closer to the mean
    #         diff = totInten[i] - meanIntenPixel[i, j]
    #         deltax[i, j] = diff * A[i, j] / M[j]
    # f1 = time.monotonic() - s1

    # totInten is a matrix (CHANNEL X CELLS)
    # meanIntenPixel is a matrix (CHANNEL X CELLS)
    # diff is a vector (diff/M[j] = scalar)
    ## diff = (totInten[i] - meanintenPixel[i,j]).dot(1/M[j])
    # diff = diff.dot(1/M)

    return totInten, meanIntenPixel, deltax


@nb.njit(parallel=True, fastmath=True)
def calc_weightedDif(weightedDif, A, totInten, meanintenPixel, ROI, ch):
    for i in nb.prange(ROI.shape[0]):
        # conditional case if totnten is 0
        if totInten[i] == 0:
            weightedDif[ch, ROI[i]] += 0
        else:
            weightedDif[ch, ROI[i]] += A[i] * abs(totInten[i] - meanintenPixel[i]) / totInten[i]


def estimatemean(
    relevantpixels, totInten, Areal, meanIntenPixel, cellArea, nonemptycellslist, chidx
):
    # meanIntenCell = [0] * (maxROI + 1)
    # weighteddif = [0] * (maxROI + 1)
    # meanIntenCell = np.zeros(maxROI + 1)
    weightedDif = np.zeros(totInten.shape)
    ROI = relevantpixels[0]

    # Areal = A[relevantpixels[0], relevantpixels[1]].toarray().reshape(-1)

    # iter on channels
    for i in range(len(meanIntenPixel)):
        meanIntenPixelreal = (
            meanIntenPixel[i][relevantpixels[0], relevantpixels[1]].toarray().reshape(-1)
        )
        totIntenreal = totInten[i][relevantpixels[0]]

        calc_weightedDif(weightedDif, Areal, totIntenreal, meanIntenPixelreal, ROI, i)

    # meanIntenCell = np.divide(totInten, cellArea, out = np.zeros_like(totInten), where= cellArea!=0)

    meanIntenCell = totInten / cellArea
    oweightedDif = np.sum(weightedDif, axis=1) / len(nonemptycellslist)
    ###############################################################
    # weighteddif = np.zeros(maxROI + 1)
    # nonemptycells = 0
    # for i in range(0, maxROI + 1):
    #     if totInten[i] > 0:
    #         nonemptycells = nonemptycells + 1
    #         # for j in nonemptypixels:
    #         for j in relevantpixels[i]:
    #             # calculate the area weighted difference between the
    #             # contributions and the goal
    #             weighteddif[i] = weighteddif[i] + A[i, j] * abs(totInten[i] - meanIntenPixel[i, j]) / totInten[i]
    #             # print('weighted difference for cell ',i,' is ',weighteddif[i])
    #             meanIntenCell[i] = totInten[i] / cellArea[i]
    #         # else:
    #         # print('Total intensity is zero for cell ',i)
    #
    # # print('weighted differences=',weighteddif)
    # overallweighteddif = sum(weighteddif) / nonemptycells

    return meanIntenCell, oweightedDif, np.amax(weightedDif, 1)


def updateX(X, maxROI, relevantpixels, drate, deltax, chidx):
    # for i in range(1, maxROI + 1):
    #     # for j in range(0,reducedsize):
    #     for j in relevantpixels[i]:
    #         X[i, j] = X[i, j] + drate * deltax[i, j]
    #         if X[i, j] < 0:
    #             X[i, j] = 0

    Xreal_CH = list(map(lambda x: x[relevantpixels[0], relevantpixels[1]], X))
    # Xreal = X[relevantpixels[0], relevantpixels[1]]

    drdeltax = list(map(lambda x: drate * x[relevantpixels[0], relevantpixels[1]], deltax))
    # drdeltax = drate * deltax[:][relevantpixels[0], relevantpixels[1]]

    a = list(map(lambda x, y: y + x, drdeltax, Xreal_CH))
    bool_CH = list(map(lambda x: find(x < 0), a))
    check_bool_CH = list(map(lambda x: len(x[0]) == 0, bool_CH))
    idx_check = [i for i, val in enumerate(check_bool_CH) if val]
    # loop - might be more efficient way
    # for i in range(len(deltax)):
    #     X[relevantpixels[0], relevantpixels] = Xreal + drdeltax[i]
    #     X_CH.append(X)

    # bool = find(X < 0)

    if idx_check:
        for i in idx_check:
            a[i][bool_CH[i][0], bool_CH[i][1]] = 0
            # X[bool[0], bool[1]] = 0

    # print(X[1,:])
    # sumX = np.sum(X, axis=0)  # sum of fractions for each pixel-should be 1's
    # print('sumsumX=',sum(sumX))
    # adjust the fractions to sum to 1 (conserve intensity)

    for i in range(len(X)):
        X[i][relevantpixels[0], relevantpixels[1]] = a[i]
        X[i] = preprocessing.normalize(X[i], norm="l1", axis=0)
        X[i] = X[i].tolil()

    # X = preprocessing.normalize(X, norm='l1', axis=0)
    # X = X.tocsr() #convert to crc for faster arthmetic
    # for j in range(0, reducedsize):
    #     xs = sum(X[:, j])
    #     X[:, j] = X[:, j] / xs.data[0]
    # X = X.tolil()

    return X


# @nb.njit(parallel=True)
# def get_indices_from_channels(Ai, Aj, M):
#
#     channel_relevantpix = nbList()
#
#     for i in nb.prange(0, M.shape[0]):
#         relevantpixels = nbList()
#
#         #indices = np.nonzero(np.isin(A_nnz[1], M_nnz[0]))[0]
#         Ai = A_nnz[0]
#         Aj = A_nnz[1]
#         relevantpixels.append(Ai)
#         relevantpixels.append(Aj)
#
#         channel_relevantpix.append(relevantpixels)


def getindexlists(cellArea: np.ndarray, A: lil_matrix):
    # nonemptycellindices = [i for i in range(0, maxROI + 1) if (cellArea[i] > 0)]
    nonemptycellindices = np.argwhere(cellArea > 0).reshape(-1)
    # print('nonemptycellindices=',nonemptycellindices)
    # emptycellindices = [i for i in range(0, maxROI) if (cellArea[i] <= 0)]
    # print('emptycellindices=',emptycellindices)
    # nonemptypixels = [j for j in range(0, reducedsize) if (M[j] > 0)]
    # print('nonemptypixels=',nonemptypixels)
    # emptypixels = [j for j in range(0, reducedsize) if (M[j] <= 0)]
    # print('emptypixels=',emptypixels)
    # relevantpixels = nbList()

    A_nnz = A.nonzero()
    # channel_relevantpix = get_indices_from_channels(A_nnz[0], A_nnz[1], M)
    A_nnz = list(A_nnz)
    # s = time.monotonic()

    #
    # M_nnz = M.nonzero()
    # indices = np.nonzero(np.isin(A_nnz[1], M_nnz[0]))[0]
    # Ai = A_nnz[0][indices]  # maxROI
    # Aj = A_nnz[1][indices]  # reducedsize
    # relevantpixels = [Ai, Aj]
    # f = time.monotonic() - s

    # s1 = time.monotonic()
    # for i in range(0, maxROI + 1):
    #     relevantpixels.append([j for j in range(0, reducedsize) if
    #                            (M[j] > 0 and A[i, j] > 0)])
    # print('relevantpixels=',relevantpixels)
    # f2 = time.monotonic() - s1
    #
    # print(f)
    # print(f2)

    return nonemptycellindices, A_nnz


def reallocateIMS(im, ROI, X, A, cellArea, reducedsize, options):
    # M   2D intensity image from IMS of size n1 x n2
    # ROI 2D mask (indexed) image of the same field as M
    # descentrate initial rate at which adjustments should be made
    #            (will be adjusted downward as necessary)
    # thresh stop when difference from desired values is less than this
    # maxiter maximum number of iterations to try
    # averagebackground average the pixels that don't have cells

    # print('IN REALLOCATE: ch-' + str(ichan) + '   time:' + str(datetime.now()))
    descentrate = options.get("reallocation_descent_rate")
    thresh = options.get("reallocation_quit_criterion")
    maxiter = options.get("num_of_reallocations")
    averagebackground = options.get("reallocation_avg_bg")
    # ROI = mask.get_data()[0, 0, 0, 0, :, :]  # assume chan 0 is the cell mask
    drate = descentrate  # set initial descent rate
    # M is all channels
    M = im.get_data()[0, 0, :, 0, :, :]
    # M = im.get_data()[0, 0, ichan, 0, :, :]

    boolnp = np.isnan(M)
    M[boolnp] = 0

    outfile = "temp"
    # plt.imshow(M)
    # plt.savefig(outfile + 'ims.png')
    # plt.imshow(ROI)
    # plt.savefig('cellmask.png')

    Mshape = M.shape  # 3D
    ROIshape = ROI.shape
    smallpixelsperbigpixel = np.prod(ROIshape) / reducedsize

    # M = np.reshape(M, -1)  # represent as vector of length n1*n2
    M = M.reshape(Mshape[0], Mshape[1] * Mshape[2])
    ROI = np.reshape(ROI, -1)  # also represent as vector

    # totalIntensity = sum(M.get_data())
    # print('sum of M=',totalIntensity)
    maxROI = max(ROI)

    # findpixelfraction outside now

    # print(X)
    # print(A)
    # print(cellArea.shape)

    print("START getindexlists..." + "   time:" + str(datetime.now()))
    nonemptycellindices, relevantpixels = getindexlists(cellArea, A)
    print("END getindexlists..." + "   time:" + str(datetime.now()))
    # Xold = lil_matrix((maxROI + 1, reducedsize))
    Xold = X
    oldweighteddif = [0] * Mshape[0]

    # find XM and prune for 0s - only need to do this once
    XM, Areal, Mreal = updateXM(X, M, A, relevantpixels)

    # not process specific channel if the threshold is surpassed
    chidx = []
    for iter in range(0, maxiter):
        print("IN ITR LOOP: " + str(iter) + "   time:" + str(datetime.now()))
        # calculate the mean intensity for the contribution of each pixel
        # to each cell
        print("START estimatedeltas..." + "   time:" + str(datetime.now()))
        totInten, meanIntenPixel, deltax = estimatedeltas(
            relevantpixels, maxROI, reducedsize, XM, M, Areal, Mreal, chidx
        )
        print("END estimatedeltas... \nSTART estimatemean..." + "   time:" + str(datetime.now()))
        meanIntenCell, overallweighteddif, maxweighteddif = estimatemean(
            relevantpixels, totInten, Areal, meanIntenPixel, cellArea, nonemptycellindices, chidx
        )

        print("END estimatemean..." + "   time:" + str(datetime.now()))
        if iter == 0:
            print("initial weighted fractional difference=", overallweighteddif)
            print("initial max weighted fractional difference=", maxweighteddif)

        maxdeltas = list(map(lambda x: np.amax(abs(x)), deltax))

        # print('maxdeltas=', maxdeltas)

        if iter == 0:
            print("START calcM..." + "   time:" + str(datetime.now()))
            # make naive estimate of contribution of intensity per cell
            initM = calcM(
                ROI,
                ROIshape,
                M,
                Mshape,
                X,
                meanIntenCell,
                averagebackground,
                smallpixelsperbigpixel,
                chidx,
            )
            print("END calcM..." + "   time:" + str(datetime.now()))
            # print('sum of initM=', sum(initM))
            # plt.imshow(initM.reshape(ROIshape))
            # plt.savefig(outfile + 'start.png')

        # check exit conditions
        # overallweighteddif - oldweighteddif < thresh?

        # find channels that have maxdeltas < thresh
        boolthresh = list(map(lambda x: x < thresh, maxdeltas))
        chidx = [i for i, val in enumerate(boolthresh) if val]

        # if (maxdeltas < thresh):
        #     # print(maxdeltas)
        #     # print(thresh)
        #     break

        bool_weighteddif = list(map(lambda x, y: x > y, overallweighteddif, oldweighteddif))
        chwidx = [i for i, val in enumerate(bool_weighteddif) if val]

        if chwidx and iter > 0:
            for i in chwidx:
                X[i] = Xold[i].copy()
                deltax[i] = olddeltax[i].copy()
                overallweighteddif[i] = oldweighteddif[i].copy()
            drate = drate / 2
            # sumX = np.sum(X, axis=0)

        # if (iter > 0 and overallweighteddif > oldweighteddif):
        #     X = Xold  # restore previous fractions
        #     deltax = olddeltax  # restore previous change estimates
        #     overallweighteddif = oldweighteddif
        #     drate = drate / 2  # decrease rate of change
        #     # print('descentrate = ',drate)
        #     sumX = np.sum(X, axis=0)  # sum of fractions for each pixel
        #     # print('sumsumX=',sum(sumX))

        # have not implemented yet
        if descentrate / drate > 100:
            # print('descent rate below minimum')
            # print(drate)
            break

        # store previous values in case the update doesn't improve
        Xold = X.copy()
        olddeltax = deltax.copy()
        oldweighteddif = overallweighteddif

        # update
        if iter < maxiter - 1:
            print("START updateX..." + "   time:" + str(datetime.now()))
            X = updateX(X, maxROI, relevantpixels, drate, deltax, chidx)
            print("END updateX..." + "   time:" + str(datetime.now()))

    if maxiter != 1:
        print("START LAST calcM..." + "   time:" + str(datetime.now()))
        newM = calcM(
            ROI,
            ROIshape,
            M,
            Mshape,
            X,
            meanIntenCell,
            averagebackground,
            smallpixelsperbigpixel,
            chidx,
        )
    # newM = np.asarray(newM, dtype=np.float32)
    # plt.imshow(newM.reshape(ROIshape))
    # plt.savefig(outfile + 'end.png')
    if iter == 0:
        newM = initM.copy()
    # else:
    #     diffM = abs(initM - newM)
    # print('maxdif=', np.amax(diffM))
    # plt.imshow(diffM.reshape(ROIshape))
    # plt.savefig(outfile + 'difference.png')

    print("final weighted fractional difference=", overallweighteddif)
    print("final max weighted fractional difference=", maxweighteddif)
    newM_CH = np.asarray(newM).reshape(((Mshape[0],) + ROIshape))
    # sumnewM = sum(newM)
    # print('sum of newM=',sumnewM,'; difference=',totalIntensity-sumnewM)

    print("FINISH REALLOCATION" + "time:" + str(datetime.now()))

    return newM_CH
