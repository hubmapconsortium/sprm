import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate, stats
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from skimage import measure
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .constants import figure_save_params

"""

Companion to SPRM.py
Package functions that are integral to running main script
Author:    Ted Zhang & Robert F. Murphy
01/21/2020 - 02/23/2020
Version: 1.00


"""


def shape_cluster(cell_matrix, typelist, all_clusters, options):
    cluster_method, min_clusters, max_clusters = options.get("num_shapeclusters")
    if max_clusters > cell_matrix.shape[0]:
        print("reducing shape clusters to ", cell_matrix.shape[0])
        num_shapeclusters = cell_matrix.shape[0]

    if cluster_method == "silhouette":
        cluster_list = []
        cluster_score = []
        for i in range(min_clusters, max_clusters + 1):
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
        cellbycluster = KMeans(n_clusters=num_shapeclusters, random_state=0).fit(cell_matrix)

    # returns a vector of len cells and the vals are the cluster numbers
    cellbycluster_labels = cellbycluster.labels_
    # print(cellbycluster_labels.shape)
    # print(cellbycluster_labels)
    # print(len(np.unique(cellbycluster_labels)))
    clustercenters = cellbycluster.cluster_centers_
    # print(clustercenters.shape)

    # save cluster info
    typelist.append("cellshapes")
    all_clusters.append(cluster_score)

    return cellbycluster_labels, clustercenters


def getcellshapefeatures(outls: np.ndarray, options: Dict) -> Tuple[np.ndarray, PCA]:
    print("Getting cell shape features...")
    numpoints = options.get("num_outlinepoints")
    # check to make sure n_components is the min of (num_outlinepoints, outls[0], outls[1])
    if options.get("num_outlinepoints") > min(outls.shape[0], outls.shape[1]):
        numpoints = min(options.get("num_outlinepoints"), outls.shape[0], outls.shape[1])

    pca_shapes = PCA(n_components=numpoints, svd_solver="full")
    # print(pca_shapes)

    #    outlinesall = outls.reshape(outls.shape[0]*outls.shape[1],outls.shape[2])
    #    print(outlinesall.shape)
    features = pca_shapes.fit_transform(outls)
    # print(features.shape)
    if features.shape[1] != numpoints:
        raise ValueError("dimensions do not match.")
    #    shape_features = features.reshape(outls.shape[0],outls.shape[1],check)

    return features, pca_shapes


def bin_pca(features, npca, cell_coord, filename, output_dir):
    sort_idx = np.argsort(features[:, npca - 1])  # from min to max
    idx = list(np.round(np.linspace(0, len(sort_idx) - 1, 11)).astype(int))
    nfeatures = features[sort_idx, 0]
    cbin = []

    for i in range(10):
        fbin = nfeatures[idx[i] : idx[i + 1]]

        # find median not mode
        median = np.median(fbin)
        # mode = stats.mode(fbin)

        nidx = np.searchsorted(fbin, median, side="left")
        r = range(idx[i], idx[i + 1])
        celln = sort_idx[r[nidx]]
        cbin.append(celln)

    f, axs = plt.subplots(1, 10)

    for i in range(10):
        cscell_coords = np.column_stack(cell_coord[cbin[i]])
        axs[i].scatter(cscell_coords[0], cscell_coords[1])
    plt.subplots_adjust(wspace=0.4)
    # plt.show()
    plt.savefig(output_dir / (filename + "-outlinePCA_bin_pca.pdf"), **figure_save_params)
    plt.close()


def pca_recon(features, npca, pca, filename, output_dir):

    # d = defaultdict(list)
    sort_idx = np.argsort(features[:, 0])  # from min to max
    idx = list(np.round(np.linspace(0, len(sort_idx) - 1, 10)).astype(int))
    idx = sort_idx[idx]

    # keep pca # same and set mode for otherse
    rfeatures = features
    samples = list(range(features.shape[1]))
    del samples[npca - 1]

    for i in samples:
        # fin median not mode
        median = np.median(rfeatures[:, i])
        # mode = stats.mode(rfeatures[:, i])
        rfeatures[:, i] = median

    # x is even y is odd
    rfeatures = pca.inverse_transform(rfeatures)

    f, axs = plt.subplots(1, 10)

    for i in range(10):
        axs[i].scatter(rfeatures[idx[i], ::2], rfeatures[idx[i], 1::2])
    plt.subplots_adjust(wspace=0.4)
    # plt.show()
    plt.savefig(output_dir / (filename + "-outlinePCA_pca_recon.pdf"), **figure_save_params)
    plt.close()


def pca_cluster_shape(features, polyg, output_dir, options):
    d = defaultdict(list)
    cell_labels, _ = shape_cluster(features, options)
    sort_idx = np.argsort(features[:, 0])

    for idx in sort_idx:
        d[cell_labels[idx]].append(polyg[idx])

    select = []
    for i in sorted(d.keys()):
        idx = list(np.round(np.linspace(0, len(d[i]) - 1, 5)).astype(int))[::-1]
        select.append(idx)

    f, axs = plt.subplots(3, 5)

    for j in range(3):
        axs[j, 0].set_title("Cluster" + str(j))

        for i in range(len(select[0])):
            # if i == 0:
            axs[j, i].scatter(d[j][select[j][i]][:, 0], d[j][select[j][i]][:, 1])
        # else:
        #     ax1.scatter(d[0][select[0][i]][:, 0] + 300, d[0][select[0][i]][:, 1])
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    # plt.show()
    plt.savefig(output_dir / "outlinePCA_cluster_pca.pdf", **figure_save_params)
    plt.close()


def create_polygons(mask, bestz: int) -> List[str]:
    """
    Adapted from Maria Keays's original create_roi_polygon's method.

    Given a NumPy ndarray mask data, the index of the best focus z-plane, create
    strings representing the polygon shapes of segmented cells and return them in a list.
    """

    # getting cells for now
    mask_img = mask.data[0, 0, 2, bestz, :, :]

    allroi = []
    for i in range(1, mask_img.max() + 1):
        # for i in range(1, 20):
        roiShape = np.where(mask_img == i)
        # roiShapeTuples = list( zip( roiShape[ 0 ], roiShape[ 1 ] ) )

        allroi.append(roiShape)

    return allroi


def cell_coord_debug(mask, nseg, npoints):
    polyg_list = []
    temp_list = []
    cellmask = mask.get_data()[0, 0, nseg, 0, :, :]

    for i in range(1, 20):
        coor = np.where(cellmask == i)

        if edgetest(coor):
            break

        cmask = np.zeros(cellmask.shape)  # comment out
        cmask[coor[1], coor[0]] = 1

        # remove this when finished
        polyg = measure.find_contours(
            cmask, 0.5, fully_connected="low", positive_orientation="low"
        )
        temp = interpalong(polyg[0], npoints)

        temp_list.append(temp)
        polyg_list.append(polyg[0])

    listofrois = create_polygons(mask, 10)

    for i in range(0, len(listofrois)):
        fig, axs = plt.subplots(1, 3)
        axs[0].set_title("Cell boundary")
        axs[0].scatter(listofrois[i][0], listofrois[i][1])

        axs[1].set_title("Sklearn")
        axs[1].scatter(polyg_list[i][:, 0], polyg_list[i][:, 1])

        axs[2].set_title("Resampling")
        axs[2].scatter(temp_list[i][:, 0], temp_list[i][:, 1])

        plt.savefig(
            f"./debug/coordinates_comparison_cell_{i + 1}",
            **figure_save_params,
        )


def getparametricoutline(mask, nseg, ROI_by_CH, options):
    print("Getting parametric outlines...")

    polygon_outlines = []
    # polygon_outlines1 = []
    cellmask = mask.get_data()[0, 0, nseg, 0, :, :]

    interiorCells = mask.get_interior_cells()

    # if options.get("num_outlinepoints") > np.amax(cellmask):
    #     options["num_outlinepoints"] = min(np.max(cellmask), options.get("num_outlinepoints"))

    npoints = options.get("num_outlinepoints")
    # the second dimension accounts for x & y coordinates for each point
    # pts = np.zeros((np.amax(cellmask), npoints * 2))
    pts = np.zeros((len(interiorCells), npoints * 2))

    cell_coords = ROI_by_CH[0]

    cell_boundary = ROI_by_CH[2]

    # for i in range(1, np.amax(cellmask)+1):
    for i in range(len(interiorCells)):
        # if i not in cellmask:
        #    continue
        # coor = np.where(cellmask == i)

        ROI_coords = cell_coords[interiorCells[i]]

        # tmask = np.zeros((cellmask.shape[1], cellmask.shape[0]))
        # tmask[ROI_coords[0], ROI_coords[1]] = 1
        #
        # polyg = measure.find_contours(tmask, 0.5, fully_connected='low', positive_orientation='low')
        # # if find contours returns back an empty array
        # if polyg[0].size == 0:
        #     polygon_outlines.append(np.zeros((npoints, 2)))
        #
        # new_array = [tuple(row) for row in polyg[0]]
        # temp = np.unique(new_array, axis=0)
        #
        # # temp = interpalong(polyg[0], npoints)
        #
        # # polygon_outlines.append(temp)
        #
        # polygon_outlines.append(temp)

        # loop to see if cell is a line - 2020

        # simple method from https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
        ptsx = ROI_coords[1] - round(np.mean(ROI_coords[1]))
        ptsy = ROI_coords[0] - round(np.mean(ROI_coords[0]))
        ptscentered = np.stack([ptsx, ptsy])
        # print(ptscentered.shape)
        xmin = min(ptscentered[0, :])
        # print(xmin)
        ymin = min(ptscentered[1, :])
        xxx = ptscentered[0, :] - xmin
        yyy = ptscentered[1, :] - ymin
        xxx = xxx.astype(int)
        yyy = yyy.astype(int)
        cmask = np.zeros(cellmask.shape)
        cmask[xxx, yyy] = 1
        # plt.imshow(cmask)
        # plt.show()

        ptscov = np.cov(ptscentered)
        # print(ptscov)

        if np.isnan(ptscov).any():
            if options.get("debug"):
                print(interiorCells[i])
                print(ptscov)
                print(ptscentered)
                cw = np.where(cellmask == interiorCells[i])
                print(cw)

                print("---")
                print(ROI_coords)
            continue

        eigenvals, eigenvecs = np.linalg.eig(ptscov)
        # print(eigenvals,eigenvecs)
        sindices = np.argsort(eigenvals)[::-1]
        # print(sindices)
        x_v1, y_v1 = eigenvecs[:, sindices[0]]  # eigenvector with largest eigenvalue
        # x_v2, y_v2 = eigenvecs[:, sindices[1]]
        theta = np.arctan((x_v1) / (y_v1))
        # print(x_v1,y_v1,theta)
        # rotationmatrix = np.matrix([[np.cos(theta), -np.sin(theta)],
        #                             [np.sin(theta), np.cos(theta)]])
        # tmat = rotationmatrix * ptscentered
        # xrotated, yrotated = tmat.A

        rotationmatrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        tmat = rotationmatrix @ ptscentered
        xrotated, yrotated = np.asarray(tmat)
        # plt.plot(xrotated,yrotated,'b+')
        # plt.show()
        # need to flip over minor axis if necessary

        tminx = min(xrotated)
        # print(tminx)
        tminy = min(yrotated)
        # print(tminy)
        xrotated = xrotated - tminx
        #        print(xrotated)
        tmatx = xrotated.round().astype(int)
        #        print(tmatx)
        yrotated = yrotated - tminy
        tmaty = yrotated.round().astype(int)

        # check skew
        x = stats.skew(tmatx)
        y = stats.skew(tmaty)

        #'heavy' end is on the left side flip to right - flipping over y axis
        if x > 0:
            tmatx = max(tmatx) - tmatx
        elif y > 0:
            tmaty = max(tmaty) - tmaty
        elif x > 0 and y > 0:
            tmatx = max(tmatx) - tmatx
            tmaty = max(tmaty) - tmaty

        # make the object mask have a border of zeroes
        cmask = np.zeros((max(tmatx) + 3, max(tmaty) + 3))
        cmask[tmatx + 1, tmaty + 1] = 1
        # fill the image to handle artifacts from rotation
        # cmask = fillimage(cmask)

        # plt.imshow(cmask)
        # plt.show()

        aligned_outline = measure.find_contours(
            cmask, 0.5, fully_connected="high", positive_orientation="low"
        )

        x = aligned_outline[0][:, 0]
        y = aligned_outline[0][:, 1]

        x, y = linear_interpolation(x, y, npoints)
        yb, xb = cell_boundary[interiorCells[i]]
        xy = np.column_stack((x, y))
        bxy = np.column_stack((xb, yb))

        # get polygons
        bxy = Polygon(bxy)
        bxy = orient(bxy)
        bxy = bxy.exterior.coords.xy
        bxy = np.array(bxy).T
        bxy = bxy.tolist()

        # find centroid
        cent = (np.sum(xb) / len(xb), np.sum(yb) / len(yb))
        bxy.sort(key=lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))
        bxy = np.array(bxy)

        polygon_outlines.append(bxy)

        flatxy = xy.reshape(-1)

        pts[i - 1, :] = flatxy

        # pts[i - 1, :] = paramshape(cmask, npoints, polyg)

    return pts, polygon_outlines


def linear_interpolation(x, y, npoints):
    points = np.array([x, y]).T  # a (nbre_points x nbre_dim) array

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)

    alpha = np.linspace(distance.min(), int(distance.max()), npoints)
    interpolator = interpolate.interp1d(distance, points, kind="slinear", axis=0)
    interpolated_points = interpolator(alpha)

    out_x = interpolated_points.T[0]
    out_y = interpolated_points.T[1]

    return out_x, out_y


def fillimage(cmask):
    changedsome = True
    while changedsome:
        changedsome = False
        for j in range(2, cmask.shape[0] - 1):
            for k in range(2, cmask.shape[1] - 1):
                if cmask[j, k] == 0:
                    m = (
                        cmask[j - 1, k - 1]
                        + cmask[j - 1, k + 1]
                        + cmask[j + 1, k - 1]
                        + cmask[j + 1, k + 1]
                    )
                    if m == 4:
                        cmask[j, k] = 1
                        changedsome = True
    return cmask


def paramshape(cellmask, npoints, polyg):
    # polyg = measure.find_contours(cellmask, 0.5, fully_connected='low', positive_orientation='low')

    #    if len(polyg) > 1:
    #        print('Warning: too many polygons found')

    polyall = np.asarray(polyg[0])

    # for i in range(0, len(polyg)):
    #     poly = np.asarray(polyg[i])
    # print(i,poly.shape)
    # print(poly[0,0],poly[0,1],poly[-1,0],poly[-1,1])
    # if i == 0:
    #     polyall = poly
    # plt.plot(poly[:,0],poly[:,1],'bo')
    # plt.plot(poly[0,0],poly[0,1],'gd')
    #        else:
    #            polyall = np.append(polyall,poly,axis=0)
    #            plt.plot(poly[:,0],poly[:,1],'rx')
    #            plt.plot(poly[0,0],poly[0,1],'y+')
    # plt.show()

    # plt.plot(polyall[:,0],polyall[:,1],'rx')
    # plt.show()

    pts = interpalong(polyall, npoints)
    # print(pts.shape)
    # plt.plot(pts[:,0],pts[:,1],'go')
    # plt.show()

    # return a linearized vector of x and y coordinates
    xandy = pts.reshape(pts.shape[0] * pts.shape[1])
    #    print(xandy)

    return xandy


def interpalong(poly, npoints):
    polylen = 0
    for i in range(0, len(poly)):
        j = i + 1
        if i == len(poly) - 1:
            j = 0
        polylen = polylen + np.sqrt(
            (poly[j, 0] - poly[i, 0]) ** 2 + (poly[j, 1] - poly[i, 1]) ** 2
        )
    # print(polylen)
    # polylen = poly.geometry().length()
    #    minlen = minneidist(poly)
    #    npmin = polylen/minlen
    #        if npmin > npoints:
    #            print('Warning: not enough interpolation points.')

    interval = polylen / npoints

    pts = np.zeros((npoints, 2))
    pts[0, :] = poly[0, :]
    j = 1
    curpos = pts[0, :]
    for i in range(1, npoints):
        sofar = 0
        while sofar < interval:
            # check whether we wrapped around
            if j >= len(poly):
                #                print('wrapped around')
                #                print(j,len(poly))
                j = 0
            # print('i,j=')
            # print(i,j)
            xdist = poly[j, 0] - curpos[0]
            ydist = poly[j, 1] - curpos[1]
            tdist = np.sqrt(xdist ** 2 + ydist ** 2)
            need = interval - sofar
            # print(xdist,ydist,tdist,need)
            if tdist >= need:
                # save next sampled position
                # print('need to interpolate')
                ocurpos = curpos.copy()
                curpos[0] = curpos[0] + (need / tdist) * xdist
                curpos[1] = curpos[1] + (need / tdist) * ydist
                # print(ocurpos,curpos)
                pts[i, :] = curpos
                sofar = interval
                #                if (curpos == poly[j,:]).all:
                if (curpos[0] == poly[j, 0]) and (curpos[1] == poly[j, 1]):
                    # print(curpos,poly[j,:])
                    # print('exact match of new point to a vertex')
                    j = j + 1
                    # print(j)
            else:
                # advance to the next vertex
                # print('advanced')
                # print(j,curpos)
                curpos = poly[j, :]
                j = j + 1
                sofar = sofar + tdist
        # print('found point')
        # print(i,j)
    # print(pts)
    return pts


# def minneidist(poly):
#    for i in range(0,poly.size[0]-1):
#        ndist = dist(poly[i,:]-poly[i+1,:])
#    return min(ndist)


def showshapesbycluster(mask, nseg, cellbycluster, filename):
    cellmask = mask.get_data()[0, 0, nseg, 0, :, :]
    # print(cellmask.shape)
    # print('Largest value in cell mask=')
    # print(np.amax(cellmask))
    #    plt.imshow(cellmask)
    #    plt.show()
    #    breakpoint()
    # print(cellbycluster)
    for k in range(0, max(cellbycluster) + 1):
        plt.figure(k + 1)
        plt.clf()
    nk = np.zeros(max(cellbycluster) + 1)
    for i in range(1, np.amax(cellmask) + 1):
        k = cellbycluster[i - 1]
        coor = np.array(np.where(cellmask == i))
        coor[0, :] = coor[0, :] - min(coor[0, :])
        coor[1, :] = coor[1, :] - min(coor[1, :])
        thisshape = np.zeros((max(coor[1, :]) + 1, max(coor[0, :]) + 1))
        thisshape[coor[1, :], coor[0, :]] = 1
        nk[k] = nk[k] + 1
        if nk[k] < 16:
            plt.figure(k + 1)
            plt.subplot(4, 4, nk[k])
            plt.imshow(thisshape)
            plt.axis("off")
        if min(nk) >= 16:
            break
    for k in range(0, max(cellbycluster) + 1):
        plt.figure(k + 1)
        plt.savefig(f"{filename}-cellshapescluster{k}.pdf", **figure_save_params)
