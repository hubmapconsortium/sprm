import io
import json
import math
import time
from bisect import bisect
from collections import ChainMap, defaultdict
from contextlib import contextmanager
from itertools import chain, combinations, product
from os import walk
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import aicsimageio
import lxml
import matplotlib
import matplotlib.cm
import matplotlib.colors
import numba as nb
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
import umap
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from apng import APNG, PNG
from matplotlib import collections as mc
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, ListedColormap, to_rgba
from numba.typed import Dict as nbDict
from numba.typed import List as nbList
from numpy import linalg as LA
from numpy import ndarray
from PIL import Image
from scipy import stats
from scipy.ndimage import binary_dilation
from scipy.spatial import KDTree

# from skimage.feature.texture import greycomatrix, greycoprops
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from .constants import FILENAMES_TO_IGNORE, INTEGER_PATTERN, figure_save_params
from .ims_sparse_allchan import findpixelfractions
from .ims_sparse_allchan import reallocateIMS as reallo
from .outlinePCA import shape_cluster

"""

Companion to SPRM.py
Package functions that are integral to running main script
Author: Ted Zhang & Robert F. Murphy
01/21/2020 - 06/25/2020
Version: 1.03


"""


@contextmanager
def new_plot():
    plt.clf()
    try:
        yield
    finally:
        plt.clf()
        plt.close()


def dataframe_pcolormesh(
    matrix: pd.DataFrame,
    cmap: Optional[Colormap] = None,
    x_label=True,
    **kwargs,
):
    """
    Plots a data frame as an image, using the specified color map to transform values.

    It is the user's responsibility to save the resulting plot.
    """
    if x_label:
        x_figsize = matrix.shape[1] * (1 / 4)
        tick_axes = "both"
    else:
        x_figsize = matrix.shape[1] / 15
        tick_axes = "x"

    plt.figure(
        figsize=(x_figsize, matrix.shape[0] / 5),
    )

    ax = plt.gca()

    ax.tick_params(axis=tick_axes, direction="out")

    if x_label:
        ax.set_xticks(range(matrix.shape[1]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(matrix.columns))
        x_minor_locator = matplotlib.ticker.AutoMinorLocator(n=2)
        x_minor_locator.MAXTICKS = 1000000000
        ax.xaxis.set_minor_locator(x_minor_locator)
        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment("center")
            tick.label1.set_rotation("vertical")

    ax.set_yticks(range(matrix.shape[0]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(matrix.index))
    y_minor_locator = matplotlib.ticker.AutoMinorLocator(n=2)
    y_minor_locator.MAXTICKS = 1000000000
    ax.yaxis.set_minor_locator(y_minor_locator)
    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_verticalalignment("center")

    if cmap is not None:
        kwargs["cmap"] = cmap
    mesh = ax.pcolormesh(matrix, **kwargs)

    plt.colorbar(mesh, ax=ax)

    ax.invert_yaxis()


def append_to_filename(path: Path, suffix: str) -> Path:
    return path.with_name(path.name + suffix)


def NMF_calc(im, fname, output_dir, options):
    """
    by Matt Ruffalo
    """

    # 2D
    channel_data = im.data[0, 0, :, 0, :, :]
    # shape_to_restore = channel_data.shape
    flattened = channel_data.reshape(
        channel_data.shape[0], np.prod(channel_data.shape[1:])
    ).transpose()

    n_components = options.get("num_channelPCA_components")

    print("Performing NMF decomposition")
    nmf = NMF(n_components=n_components)

    flattened_copy = flattened.copy()
    tries = 0
    while True:
        try:
            transformed_flat = nmf.fit(flattened_copy)
            break
        except Exception as e:
            print(e)
            print("Exceptions caught: ", tries)

            if tries == 0:
                nmf = NMF(n_components=n_components, init="random")
                tries += 1
            else:
                print("halving the dataset...")
                n_samples = int(flattened_copy.shape[0] / 2)
                idx = np.random.choice(flattened.shape[0], n_samples, replace=False)
                flattened_copy = flattened[idx, :]
                tries += 1

    transformed_flat = transformed_flat.transform(flattened)

    component_df = pd.DataFrame(
        nmf.components_.T,
        index=im.get_channel_labels(),
        columns=[f"Comp. {i}" for i in range(n_components)],
    )
    output_component_csv = output_dir / (fname + "_nmf_components.csv")
    print("Saving NMF components CSV to", output_component_csv)
    component_df.to_csv(output_component_csv)

    output_component_pdf = output_dir / (fname + "_nmf_components.pdf")
    print("Saving NMF components PDF to", output_component_pdf)
    with new_plot():
        dataframe_pcolormesh(component_df)
        plt.savefig(output_component_pdf, bbox_inches="tight")

    print("Reshaping data to save as image")
    transformed = transformed_flat.transpose().reshape(n_components, *channel_data.shape[1:])
    transformed_plot = np.transpose(transformed, (1, 2, 0))
    print("Normalizing transformed values")
    for i in range(n_components):
        transformed_plot[..., i] *= 255.0 / transformed_plot[..., i].max()

    transformed_plot = transformed_plot.round().astype(np.uint8)
    output_png = output_dir / (fname + "_nmf_top3.png")
    print("Saving NMF-transformed image to", output_png)
    # save_image(transformed_plot, output_png, options)
    img = Image.fromarray(transformed_plot, mode="RGB")
    img.save(output_png)


# for aicsimageio<4
def get_channel_names(image: aicsimageio.AICSImage) -> List[str]:
    if isinstance(image.metadata, aicsimageio.vendor.omexml.OMEXML):
        # aicsimageio<4 doesn't allow overriding whether to include
        # an encoding declaration or to skip decoding to text from bytes,
        # so re-encode to UTF-8 bytes ourselves
        metadata = image.metadata.to_xml().encode("utf-8")
    elif isinstance(image.metadata, str):
        # shouldn't have an encoding declaration if aicsimageio didn't
        # recognize it as OME-XML, so lxml needs it to be bytes
        metadata = image.metadata
    tree = lxml.etree.fromstring(metadata)
    namespaces = tree.nsmap.copy()
    namespaces["OME"] = namespaces[None]
    del namespaces[None]
    channels = tree.xpath("//OME:Pixels/OME:Channel/@Name", namespaces=namespaces)
    return channels


class IMGstruct:
    """
    Main Struct for IMG information
    """

    img: AICSImage
    data: np.ndarray
    path: Path
    name: str

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

    # def quit(self):
    #     self.img = None
    #     self.data = None
    #     self.path = None
    #     self.name = None
    #     self.channel_labels = None

    @staticmethod
    def read_img(path: Path, options: Dict) -> AICSImage:
        img = AICSImage(path)
        if not img.metadata:
            print("Metadata not found in input image")
            # might be a case-by-basis
            # img = AICSImage(path), known_dims="CYX")

        return img

    def read_data(self, options):
        data = self.img.data
        dims = data.shape

        # Haoran: hot fix for 5 dims 3D IMC images
        # if len(self.img.data.shape) == 5:
        #     data = self.img.data[:, :, :, :, :]
        #     dims = data.shape
        #     s, c, z, y, x = dims[0], dims[1], dims[2], dims[3], dims[4]
        # else:
        #     data = self.img.data[:, :, :, :, :, :]
        #     dims = data.shape
        #     s, t, c, z, y, x = dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
        #     if t > 1:
        #         data = data.reshape((s, 1, t * c, z, y, x))

        # older version of aicsimageio<3.2
        if len(dims) == 6:
            s, t, c, z, y, x = dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
            if t > 1:
                data = data.reshape((s, 1, t * c, z, y, x))

        # newer version of aicsimageio>4.0 -> correct dimensions
        elif len(dims) == 5:
            data = data[np.newaxis, ...]

        else:
            print(
                "image/expressions dimensions are incompatible. Please check that its in correct format."
            )

        # convert data to a float for normalization downstream
        data = data.astype("float")
        assert data.dtype == "float"

        return data

    def read_channel_names(self):
        img: AICSImage = self.img
        # cn = get_channel_names(img)
        cn = img.channel_names
        print("Channel names:")
        print(cn)

        # hot fix to channel names expected
        if cn[0] == "cells":
            cn[0] = "cell"
        if cn[2] == "cell_boundary":
            cn[2] = "cell_boundaries"
        if cn[3] == "nucleus_boundary":
            cn[3] = "nucleus_boundaries"

        return cn

    def get_name(self):
        return self.name

    def get_channel_labels(self):
        return self.channel_labels

    def set_channel_labels(self, channel_list):
        self.channel_labels = channel_list


class MaskStruct(IMGstruct):
    """
    Main structure for segmentation information
    """

    def __init__(self, path: Path, options):
        super().__init__(path, options)
        self.bestz = self.get_bestz()
        self.interior_cells = []
        self.edge_cells = []
        self.cell_index = []
        self.bad_cells = []
        self.ROI = []

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

        # aicsimageio > 4.0
        if len(dims) == 5:
            data = data[np.newaxis, ...]

        check = data[:, :, :, 0, :, :]
        check_sum = np.sum(check)
        if check_sum == 0:  # assumes the best z is not the first slice
            print("Duplicating best z to all z dimensions...")
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

            if options.get("debug"):
                print("Best z dimension found: ", bestz)
            # data now contains just the submatrix that has nonzero values
            # add back the z dimension
            data = x[:, :, :, np.newaxis, :, :]
            # print(data.shape)
            # and replicate it
            data = np.repeat(data, dims[3], axis=3)
            # print(data.shape)
            # set bestz
        else:  # 3D case or that slice 0 is best
            if dims[2] > 1:
                bestz.append(int(data.shape[3] / 2))
            else:
                bestz.append(0)

        # set bestz
        self.set_bestz(bestz)

        # check to make sure is int64
        data = data.astype(int)
        assert data.dtype == "int"

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

    def set_ROI(self, ROI):
        self.ROI = ROI

    def get_ROI(self):
        return self.ROI

    # def quit(self):
    #     self.img = None
    #     self.data = None
    #     self.path = None
    #     self.name = None
    #     self.channel_labels = None
    #     self.bestz = None
    #     self.interior_cells = None
    #     self.edge_cells = None
    #     self.bad_cells = None
    #     self.cell_index = None
    #     self.ROI = None


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


class Features:
    """
    Structure for all features --- will be implemented in the future
    """

    def __init__(self):
        self.features = defaultdict()


def create_custom_cmap(n_colors=10):
    # Define first three colors
    color_names = ["red", "green", "blue"]

    # Convert color names to RGBA
    colors = [to_rgba(c) for c in color_names]

    # Add other colors from the 'hsv' colormap
    hsv = plt.cm.get_cmap("hsv", n_colors)

    for i in range(3, n_colors):
        colors.append(hsv(i))  # Get RGB color and add to list

    # Create and return custom colormap
    # cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_colors)
    cmap = ListedColormap(colors, name="custom_cmap", N=n_colors)
    return cmap


def choose_colormap_new(a: np.ndarray) -> np.ndarray:
    # max_value = int(a.max() - 1)
    cmap = create_custom_cmap()

    return np.array(cmap.colors)[:, :3]


colormap_choices = ["Set1", "tab20"]
unlimited_colormap = "gist_rainbow"
colormap_lengths = [len(matplotlib.cm.get_cmap(c).colors) for c in colormap_choices]


def choose_colormap(a: np.ndarray, options: Dict) -> np.ndarray:
    """
    Choose a color map for an integer-valued array denoting categorical
    values for each pixel. Returns a np.ndarray with three columns corresponding
    to RGB channels, with either as many rows as in a categorical color map,
    or as many rows as required to represent all values of `a` when adding
    dark gray as color 0.

    Uses the first color map in `colormap_choices` that fits all categorical
    values, or falls back to `unlimited colormap`.

    >>> a1 = np.arange(5)
    >>> a1_cmap = choose_colormap(a1)
    >>> (a1_cmap == np.array(matplotlib.cm.get_cmap('Set1').colors)).all()
    True
    >>> a2 = np.arange(16)
    >>> a2_cmap = choose_colormap(a2)
    >>> (a2_cmap == np.array(matplotlib.cm.get_cmap('tab20').colors)).all()
    True
    >>> a3 = np.arange(25)
    >>> a3_cmap = choose_colormap(a3)
    >>> a3_cmap.shape
    (24, 3)
    """
    # We add dark gray as the first color, so we can handle one more
    # color than the original color map -- handle this by subtracting
    # 1 from the maximum categorical value
    if "max_cluster" in options:
        max_value = options.get("max_cluster") - 1
    else:
        max_value = a.max() - 1

    choice = bisect(colormap_lengths, max_value)

    if choice in range(len(colormap_choices)):
        cmap = matplotlib.cm.get_cmap(colormap_choices[choice])
        return np.array(cmap.colors)

    # else fall back to unlimited colormap
    norm = matplotlib.colors.Normalize(vmin=a.min(), vmax=max_value, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=unlimited_colormap)
    # Drop alpha channel; just want RGB from this function
    colors = mapper.to_rgba(np.arange(a.max()))[:, :3]
    return colors


def adjust_matplotlib_categorical_cmap(
    cmap: np.ndarray,
    zero_color: float = 0.125,
    zero_alpha: float = 1.0,
) -> np.ndarray:
    colors = np.vstack([np.repeat(zero_color, 3), cmap])
    colors = np.hstack([colors, np.ones((colors.shape[0], 1))])
    colors[0, 3] = zero_alpha
    return colors


def save_image(
    a: np.ndarray,
    file_path: Union[str, Path],
    options: dict,
):
    """
    :param a: 2 or 3-dimensional NumPy array
    """
    if (a.round() != a).any():
        raise ValueError("need integral values for categorical plots")
    a = a.astype(np.uint)

    # cmap = choose_colormap_new(a)
    cmap = choose_colormap(a, options)
    adjusted_cmap = adjust_matplotlib_categorical_cmap(cmap)
    image_rgb = adjusted_cmap[a]
    image_rgb_8bit = (image_rgb * 255).round().astype(np.uint8)

    # find out if input image is 2D or 3D
    if image_rgb_8bit.ndim <= 3:
        i = Image.fromarray(image_rgb_8bit, mode="RGBA")
        i.save(file_path)
    else:  # image is 3D and will use animated png to display
        apng = APNG()
        # forward
        for j in range(len(image_rgb_8bit)):
            in_mem_file = io.BytesIO()
            i = Image.fromarray(image_rgb_8bit[j], mode="RGBA")
            i.save(in_mem_file, format="PNG")
            in_mem_file.seek(0)
            img_bytes = in_mem_file.read()
            png = PNG.from_bytes(img_bytes)
            apng.append(png, delay=options.get("apng_delay"))

        # backward
        for j in range(len(image_rgb_8bit)):
            in_mem_file = io.BytesIO()
            i = Image.fromarray(image_rgb_8bit[-j], mode="RGBA")
            i.save(in_mem_file, format="PNG")
            in_mem_file.seek(0)
            img_bytes = in_mem_file.read()
            png = PNG.from_bytes(img_bytes)
            apng.append(png, delay=options.get("apng_delay"))

        apng.save(file_path)


def calculations(
    coord, im: IMGstruct, t: int, i: int, bestz: int
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Returns covariance matrix, mean vector, and total vector
    """

    if i == 0:
        print("Performing statistical analyses on ROIs...")

    if coord.shape[0] > 2:
        z, y, x = coord[0], coord[1], coord[2]
        z = z.astype(int)
    else:
        y, x = coord[0], coord[1]
        z = bestz

    y = y.astype(int)
    x = x.astype(int)

    temp = im.get_data()

    channel_all_mask = temp[0, t, :, z, y, x]
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
    """
    Receives out_matrix and extracts information to output a vector:
    len # of cells with corresponding cluster number
    """
    # extracting all cells through all timepoints --> 4D matrix
    # cell_matrix = outmatrix[:,0,:,:,:]
    # dims = get_dims(cell_matrix)
    # t, cl, cha, chb  = dims[0], dims[1], dims[2], dims[3]
    if cell_matrix.shape[0] > 1:
        raise NotImplementedError("cell matrix with > 1 time point is not supported yet")
    cell_matrix = cell_matrix[0, :, :, :]
    # optional: pull out one seg method
    if segnum >= 0:
        cell_matrix = cell_matrix[segnum, :, :, :]
        cell_matrix = cell_matrix.reshape(
            (cell_matrix.shape[0], cell_matrix.shape[1] * cell_matrix.shape[2])
        )
        if options.get("debug"):
            print("segmentation channel: " + str(segnum + 1))
            print("3D cell matrix")
            print(cell_matrix.shape)
    else:
        # temp_matrix = np.zeros([cell_matrix.shape[1], cell_matrix.shape[2], cell_matrix.shape[3], cell_matrix.shape[0]])
        # for i in range(0, cell_matrix.shape[0]):
        #     temp_matrix[:, :, :, i] = cell_matrix[i, :, :, :]
        # cell_matrix2 = temp_matrix
        # cell_matrix2 = cell_matrix2.reshape(
        #     (cell_matrix2.shape[0], cell_matrix2.shape[1] * cell_matrix2.shape[2] * cell_matrix2.shape[3]))

        cell_matrix = np.squeeze(cell_matrix)
        cell_matrix = np.concatenate(cell_matrix, axis=1)

        if len(cell_matrix.shape) > 2:
            cell_matrix = cell_matrix.reshape(cell_matrix.shape[0], -1)
        # cell_matrix = cell_matrix.reshape(
        #     (cell_matrix.shape[1], cell_matrix.shape[2] * cell_matrix.shape[3] * cell_matrix.shape[0]))

        # cell_matrix =
    return cell_matrix


def cell_cluster(
    cell_matrix: np.ndarray,
    typelist: List,
    all_clusters: List,
    s: str,
    options: Dict,
    output_dir: Path,
    inCells: list,
) -> np.ndarray:
    # kmeans clustering
    print("Clustering cells...")
    # check of clusters vs. n_sample wanted
    cluster_method, min_cluster, max_cluster, keep = options.get("num_cellclusters")
    if max_cluster > cell_matrix.shape[0]:
        print("reducing cell clusters to ", cell_matrix.shape[0])
        num_cellclusters = cell_matrix.shape[0]

    # skipping clustering because all 0s of texture
    if options.get("texture_flag"):
        cellbycluster = KMeans(n_clusters=1, random_state=0).fit(cell_matrix)
        options.pop("texture_flag", None)
        cluster_score = []
    else:
        if cluster_method == "silhouette":
            cluster_list = []
            cluster_score = []
            for i in range(min_cluster, max_cluster + 1):
                # cellbycluster = KMeans(n_clusters=i, random_state=0)
                cellbycluster = KMeans(n_clusters=i, random_state=0, tol=1e-6)
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

    # sort the clusters by the largest to smallest and then reindex
    # unique_clusters = set(cellbycluster_labels)
    # cluster_counts = {cluster: (cellbycluster_labels == cluster).sum() for cluster in unique_clusters}
    # sorted_clusters = sorted(cluster_counts, key=cluster_counts.get, reverse=True)
    #
    # lut = np.zeros_like(cellbycluster_labels)
    # lut[sorted_clusters] = np.arange(len(unique_clusters))

    # save score and type it was
    typelist.append(s)
    all_clusters.append(cluster_score)

    # write out that cluster ids
    if keep and "cluster_list" in locals():
        all_labels = [x.labels_ for x in cluster_list]
        df = pd.DataFrame(all_labels).T
        df.columns = [s + ":" + str(i) for i in range(min_cluster, max_cluster + 1)]
        df.to_csv(output_dir / (s + "_all_clusters.csv"))

    return cellbycluster_labels, clustercenters


# def map: cell index to cluster index. Mask indexed img changes that to cluster number (mask,cellbycluster)
def cell_map(mask: MaskStruct, cc_v: pd.DataFrame, seg_n: int, options: Dict) -> List[np.ndarray]:
    """
    Maps the cells to indexed img
    """

    mask_img = mask.get_data()
    mask_img = mask_img[0, 0, seg_n, :, :, :]
    # temp = np.zeros(mask_img.shape)

    inCells = np.asarray(mask.get_interior_cells())

    list_of_cluster_imgs = []
    # loop through all features or columns
    print("Mapping...")
    stime = time.monotonic() if options.get("debug") else None
    for i in cc_v.columns:
        # for now skip cell types until one hot encoded
        if i == "DeepCell Type":
            continue
        temp = np.zeros(mask_img.shape)  # reinitiate the mask
        feature = cc_v.loc[:, i].to_numpy()
        clusters = np.unique(feature)

        for j in range(0, len(clusters)):
            cell_num = np.where(feature == clusters[j])[0]
            cell_num = inCells[cell_num]
            bit_mask = np.isin(mask_img, cell_num)
            temp[bit_mask] = clusters[j]

        list_of_cluster_imgs.append(temp)

    if options.get("debug"):
        print("Elapsed time for cell mapping <vectorized>: ", time.monotonic() - stime)

    return list_of_cluster_imgs


# @nb.njit(parallel=True)
def append_coord(masked_imgs_coord, rlabel_mask, indices):
    for i in range(0, len(rlabel_mask)):
        masked_imgs_coord[rlabel_mask[i]][0].append(indices[0][i])
        masked_imgs_coord[rlabel_mask[i]][1].append(indices[1][i])

    return


def append_coord_3D(masked_imgs_coord, rlabel_mask, indices):
    for i in range(0, len(rlabel_mask)):
        masked_imgs_coord[rlabel_mask[i]][0].append(indices[0][i])
        masked_imgs_coord[rlabel_mask[i]][1].append(indices[1][i])
        masked_imgs_coord[rlabel_mask[i]][2].append(indices[2][i])

    return


@nb.njit(parallel=True)
def cell_num_index_map(flat_mask: np.ndarray, cell_num_dict: Dict):
    for i in nb.prange(0, len(flat_mask)):
        flat_mask[i] = cell_num_dict.get(flat_mask[i])


def unravel_indices_3D(mask_channels, maxvalue, channel_coords):
    for j in range(0, len(mask_channels)):
        # might want to change this to pure numpy arrays
        masked_imgs_coord = [[[], [], []] for i in range(maxvalue)]
        labeled_mask = mask_channels[j]
        rlabel_mask = labeled_mask[:, :, :].reshape(-1)
        indices = np.arange(len(rlabel_mask))
        indices = np.unravel_index(
            indices, (labeled_mask.shape[0], labeled_mask.shape[1], labeled_mask.shape[2])
        )

        append_coord_3D(masked_imgs_coord, rlabel_mask, indices)
        masked_imgs_coord = list(map(np.asarray, masked_imgs_coord))

        channel_coords.append(masked_imgs_coord)


def unravel_indices(mask_channels, maxvalue, channel_coords):
    for j in range(0, len(mask_channels)):
        # might want to change this to pure numpy arrays
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
    mask.set_cell_index(cell_num[1:])

    if maxvalue - 1 != np.max(mask_data):
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

        cell_num = np.unique(mask_data)
        maxvalue = len(cell_num)

    assert (maxvalue - 1) == np.max(mask_data)

    # post-process for edge case cell coordinates - only 1 point
    freq = np.unique(mask_data[0, 0, 0, :, :, :], return_counts=True)
    idx = np.where(freq[1] < options.get("valid_cell_threshold"))[0].tolist()
    mask.set_bad_cells(idx)

    mask_4D = mask_data[0, 0, :, :, :, :]

    for i in range(0, mask_4D.shape[0]):
        mask_channels.append(mask_4D[i, :, :, :])

    # 3D case
    if mask_4D.shape[1] > 1:
        unravel_indices_3D(mask_channels, maxvalue, channel_coords)
    else:
        unravel_indices(mask_channels, maxvalue, channel_coords)  # new
    # npwhere(mask_channels, maxvalue, channel_coords_np) #old

    # remove idx from coords
    # if len(idx) != 0:
    #     for i in range(len(channel_coords)):
    #         channel_coords[i] = [i for j, i in enumerate(channel_coords[i]) if j not in idx]

    return channel_coords


def cell_graphs(
    mask: MaskStruct,
    ROI_coords: List[List[np.ndarray]],
    inCells: List,
    fname: str,
    outputdir: Path,
    options: Dict,
):
    """
    Get cell centers as well as adj list of cells
    """

    cellmask = ROI_coords[0]
    cell_center = np.zeros((len(cellmask), 3))
    # cell_idx = mask.get_cell_index()

    if cellmask[0].shape[0] == 3:
        for i in range(1, len(cellmask)):
            # m = (np.sum(cellmask[i], axis=1) / cellmask[i].shape[1]).astype(int)
            m = np.mean(cellmask[i], axis=1).astype(int)
            cell_center[i, 0] = m[1]
            cell_center[i, 1] = m[2]
            cell_center[i, 2] = m[0]
    else:
        for i in range(1, len(cellmask)):
            # m = (np.sum(cellmask[i], axis=1) / cellmask[i].shape[1]).astype(int)
            m = np.mean(cellmask[i], axis=1).astype(int)
            cell_center[i, 0] = m[0]
            cell_center[i, 1] = m[1]

    cell_center_df = pd.DataFrame(cell_center)
    cell_center_df.index.name = "ID"

    cell_center_df.to_csv(outputdir / (fname + "-cell_centers.csv"), header=["x", "y", "z"])
    adj_cell_list(mask, ROI_coords, cell_center_df, inCells, fname, outputdir, options)

    # adj_cell_list(cellmask, fname, outputdir)

    # return cell_center


def adj_cell_list(
    mask,
    ROI_coords: List[np.ndarray],
    cell_center: np.ndarray,
    inCells: list,
    fname: str,
    outputdir: Path,
    options: Dict,
):
    """
    Construct adj list of neighboring cells
    """
    print("Finding cell adjacency matrix...")

    # mask_data = mask.get_data()[0, 0, 3, 0, :, :]
    # interiorCells = mask.get_interior_cells()
    stime = time.monotonic()

    if options.get("cell_graph") == 1:
        if ROI_coords[2][0].shape[0] == 2:
            AdjacencyMatrix(mask, ROI_coords, cell_center, inCells, fname, outputdir, options)
            # adjmatrix = AdjacencyMatrix(mask_data, edgecoords, interiorCells)
        else:  # 3D
            AdjacencyMatrix_3D(mask, ROI_coords, cell_center, inCells, fname, outputdir, options)
        print("Runtime of adj matrix: ", time.monotonic() - stime)
    else:
        df = pd.DataFrame(np.zeros(1))
        df.to_csv(outputdir / (fname + "-cell_adj_list.csv"))


@nb.njit(parallel=True)
def nb_CheckAdjacency(cellCoords_control, cellCoords_cur, thr):
    minValue = np.inf

    for k in nb.prange(len(cellCoords_cur[0])):
        l = nbList()
        sub = cellCoords_control[0] - cellCoords_cur[0, k]
        sub2 = cellCoords_control[1] - cellCoords_cur[1, k]
        # subtracted = np.stack((sub, sub2))
        for j in nb.prange(len(sub)):
            a = np.array([sub[j], sub2[j]], dtype=np.float32)
            norm = LA.norm(a)
            l.append(norm)
        # distance = np.array(l)
        # subtracted = np.asarray([cellCoords_control[0] - cellCoords_cur[0, k], cellCoords_control[1] - cellCoords_cur[1, k]])
        # distance = LA.norm(subtracted, axis=0)
        # d = LA.norm(subtracted)
        if min(l) < thr:
            return min(l)

    return 0


def cell_check(cc, cell):
    d = {}
    d[cell.tobytes()] = None
    if cc.tobytes() in d.keys():
        return True
    else:
        return False


def CheckAdjacency_Distance(cell_center, cell, cellids, idx, adjmatrix, cellGraph, i):
    for j in cellids[i]:
        # if j >= idx[-1]:
        #     continue

        k = idx[i]

        # check for edge cases in which cell centers lie within another cell
        bool = cell_check(cell_center[j], cell[k])
        boole = cell_check(cell_center[k], cell[j])

        if bool or boole:
            continue
        else:
            sub = cell_center[k] - cell_center[j]
            distance = LA.norm(sub)
            adjmatrix[k, j] = distance
            adjmatrix[j, k] = distance
            cellGraph[k].add(j)
            cellGraph[j].add(k)


def CheckAdjacency_Distance_new(celledge, cellids, idx, adjmatrix, cellGraph, i):
    for j in cellids[i]:
        # if j >= idx[-1]:
        #     continue

        k = idx[i]

        a = np.asarray(celledge[j])
        b = np.asarray(celledge[k])
        tree = KDTree(a.T)
        dist, _ = tree.query(b.T, k=[1])  # k desired number of neighbors
        # res_df = df[["X1", "Y1"]]
        # res_df[["X2", "Y2"]] = df[["X2", "Y2"]].iloc[ind].reset_index(drop=True)
        # res_df["distance"] = dist
        # sub = cell_center[k] - cell_center[j]

        # distance = LA.norm(sub)
        distance = np.argmin(dist)

        adjmatrix[k, j] = distance
        adjmatrix[j, k] = distance
        cellGraph[k].add(j)
        cellGraph[j].add(k)


def AdjacencyMatrix_3D(
    mask,
    ROI_coords,
    cell_center: pd.DataFrame,
    inCells: list,
    baseoutputfilename,
    output_dir,
    options: Dict,
    window=None,
):
    """
    By: Young Je Lee, Ted Zhang and Matt Ruffalo
    """

    ###
    # change from list[np.arrays] -> np.array
    ###
    # cel = nbList()
    thr = options.get("cell_adj_dilation_itr")
    cell_center = cell_center.to_numpy()

    paraopt = options.get("cell_adj_parallel")
    loc = mask.get_labels("cell_boundaries")
    print("adjacency calculation begin")
    # start = time.perf_counter()

    cellEdgeList = ROI_coords[2]

    # numCells = len(inCells)
    numCells = len(cellEdgeList)
    # cellidx = mask.get_cell_index()
    # intCells = mask.get_interior_cells()
    # assert (numCells == intCells)

    # min_dist
    adjacencyMatrix = scipy.sparse.dok_matrix((numCells, numCells))
    cellGraph = defaultdict(set)

    # avg dist
    adjmatrix_avg = scipy.sparse.dok_matrix((numCells, numCells))
    cellGraph_avg = defaultdict(set)

    if window is None:
        delta = options.get("adj_matrix_delta")
    else:
        delta = len(window) + options.get("adj_matrix_delta")

    # maskImg = mask.get_data()[0, 0, loc, 0, :, :]
    maskImg = mask.get_data()[0, 0, loc, :, :, :]

    z = maskImg.shape[0]
    a = maskImg.shape[2]
    b = maskImg.shape[1]

    if paraopt == 1:
        cel = nbList(cellEdgeList)
        # not compatible for 3D
        windowCoords, windowSize, windowXY = nbget_windows(numCells, cel, inCells, delta, a, b)
    else:
        windowCoords, windowSize, windowXY = get_windows_3D(numCells, cellEdgeList, delta, a, b, z)

    templist = []
    for i in range(len(windowCoords)):
        tempImg = np.zeros((windowSize[i][2], windowSize[i][1], windowSize[i][0]))
        tempImg[windowCoords[i][2, :], windowCoords[i][1, :], windowCoords[i][0, :]] = 1
        templist.append(tempImg)

    dimglist = [binary_dilation(x, iterations=thr) for x in templist]
    maskcrop = [maskImg[x[4] : x[5], x[2] : x[3], x[0] : x[1]] for x in windowXY]
    nimglist = [x * y for x, y in zip(maskcrop, dimglist)]
    cellids = [np.unique(x)[1:] for x in nimglist]
    idx = np.arange(1, len(cellids) + 1)
    cellids = [np.delete(x, x == y) for x, y in zip(cellids, idx)]

    for i in range(len(cellids)):
        if cellids[i].size == 0:
            continue
        else:
            CheckAdjacency_Distance(
                cell_center,
                ROI_coords[0],
                cellids,
                idx,
                adjmatrix_avg,
                cellGraph_avg,
                i,
            )

            # for min dist
            CheckAdjacency_Distance_new(
                cellEdgeList,
                cellids,
                idx,
                adjacencyMatrix,
                cellGraph,
                i,
            )

    AdjacencyMatrix2Graph(
        adjacencyMatrix,
        cell_center,
        cellGraph,
        output_dir / (baseoutputfilename + "_AdjacencyGraph.pdf"),
        thr,
    )
    # Remove background
    adjacencyMatrix = adjacencyMatrix[1:, 1:]
    adjmatrix_avg = adjmatrix_avg[1:, 1:]

    adjmatrix_avg_csr = scipy.sparse.csr_matrix(adjmatrix_avg)
    adjacencyMatrix_csr = scipy.sparse.csr_matrix(adjacencyMatrix)
    scipy.io.mmwrite(
        output_dir / (baseoutputfilename + "_AdjacencyMatrix.mtx"), adjacencyMatrix_csr
    )
    scipy.io.mmwrite(
        output_dir / (baseoutputfilename + "_AdjacencyMatrix_avg.mtx"), adjmatrix_avg_csr
    )
    with open(output_dir / (baseoutputfilename + "_AdjacencyMatrixRowColLabels.txt"), "w") as f:
        for i in range(numCells):
            print(i, file=f)


def AdjacencyMatrix(
    mask,
    ROI_coords,
    cell_center: pd.DataFrame,
    inCells: list,
    baseoutputfilename,
    output_dir,
    options: Dict,
    window=None,
):
    """
    By: Young Je Lee, Ted Zhang and Matt Ruffalo
    """

    ###
    # change from list[np.arrays] -> np.array
    ###
    # cel = nbList()
    thr = options.get("cell_adj_dilation_itr")
    cell_center = cell_center.to_numpy()

    paraopt = options.get("cell_adj_parallel")
    loc = mask.get_labels("cell_boundaries")
    print("adjacency calculation begin")
    # start = time.perf_counter()

    cellEdgeList = ROI_coords[2]

    # numCells = len(inCells)
    numCells = len(cellEdgeList)
    cellidx = mask.get_cell_index()
    # intCells = mask.get_interior_cells()
    # assert (numCells == intCells)

    # min_dist
    adjacencyMatrix = scipy.sparse.dok_matrix((numCells, numCells))
    cellGraph = defaultdict(set)

    # avg dist
    adjmatrix_avg = scipy.sparse.dok_matrix((numCells, numCells))
    cellGraph_avg = defaultdict(set)

    if window is None:
        delta = options.get("adj_matrix_delta")
    else:
        delta = len(window) + options.get("adj_matrix_delta")

    maskImg = mask.get_data()[0, 0, loc, 0, :, :]
    # maskImg = mask.get_data()

    # #check if 3D or not
    # if maskImg.shape[3] > 1:
    #     z = maskImg.shape[3]

    a = maskImg.shape[1]
    b = maskImg.shape[0]

    if paraopt == 1:
        cel = nbList(cellEdgeList)
        windowCoords, windowSize, windowXY = nbget_windows(numCells, cel, inCells, delta, a, b)
    else:
        windowCoords, windowSize, windowXY = get_windows(numCells, cellEdgeList, delta, a, b)

    templist = []
    for i in range(len(windowCoords)):
        tempImg = np.zeros((windowSize[i][1], windowSize[i][0]))
        tempImg[windowCoords[i][1, :], windowCoords[i][0, :]] = 1
        templist.append(tempImg)

    dimglist = [binary_dilation(x, iterations=thr) for x in templist]
    maskcrop = [maskImg[x[2] : x[3], x[0] : x[1]] for x in windowXY]
    nimglist = [x * y for x, y in zip(maskcrop, dimglist)]
    cellids = [np.unique(x)[1:] for x in nimglist]
    idx = np.arange(1, len(cellids) + 1)
    cellids = [np.delete(x, x == y) for x, y in zip(cellids, idx)]

    for i in range(len(cellids)):
        if cellids[i].size == 0:
            continue
        else:
            # for avg dist
            CheckAdjacency_Distance(
                cell_center,
                ROI_coords[0],
                cellids,
                idx,
                adjmatrix_avg,
                cellGraph_avg,
                i,
            )

            # for min dist
            CheckAdjacency_Distance_new(
                cellEdgeList,
                cellids,
                idx,
                adjacencyMatrix,
                cellGraph,
                i,
            )

    AdjacencyMatrix2Graph(
        adjacencyMatrix,
        cell_center,
        cellGraph,
        output_dir / (baseoutputfilename + "_AdjacencyGraph.pdf"),
        thr,
    )
    # Remove background
    adjacencyMatrix = adjacencyMatrix[1:, 1:]
    adjmatrix_avg = adjmatrix_avg[1:, 1:]

    adjmatrix_avg_csr = scipy.sparse.csr_matrix(adjmatrix_avg)
    adjacencyMatrix_csr = scipy.sparse.csr_matrix(adjacencyMatrix)
    scipy.io.mmwrite(
        output_dir / (baseoutputfilename + "_AdjacencyMatrix.mtx"), adjacencyMatrix_csr
    )
    scipy.io.mmwrite(
        output_dir / (baseoutputfilename + "_AdjacencyMatrix_avg.mtx"), adjmatrix_avg_csr
    )
    with open(output_dir / (baseoutputfilename + "_AdjacencyMatrixRowColLabels.txt"), "w") as f:
        for i in range(numCells):
            print(i, file=f)

    # return adjacencyMatrix


def get_windows_3D(numCells, cellEdgeList, delta, a, b, c):
    windowCoords = []
    windowSize = []
    windowRange_xy = []

    for i in range(1, numCells):
        # get window ranges
        xmin, xmax, ymin, ymax, zmin, zmax = (
            np.min(cellEdgeList[i][2]),
            np.max(cellEdgeList[i][2]),
            np.min(cellEdgeList[i][1]),
            np.max(cellEdgeList[i][1]),
            np.min(cellEdgeList[i][0]),
            np.max(cellEdgeList[i][0]),
        )

        # padding
        xmin = xmin - delta if xmin - delta > 0 else 0
        xmax = xmax + delta + 1 if xmax + delta + 1 < a else a
        ymin = ymin - delta if ymin - delta > 0 else 0
        ymax = ymax + delta + 1 if ymax + delta + 1 < b else b
        zmin = zmin - delta if zmin - delta > 0 else 0
        zmax = zmax + delta + 1 if zmax + delta + 1 < c else c
        # tempImg = np.zeros((xmax - xmin, ymax - ymin))
        xyz = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
        windowRange_xy.append(xyz)

        x = xmax - xmin
        y = ymax - ymin
        z = zmax - zmin
        d = np.array([x, y, z])

        temp1 = cellEdgeList[i][2] - xmin
        temp2 = cellEdgeList[i][1] - ymin
        temp3 = cellEdgeList[i][0] - zmin

        temp = np.stack((temp1, temp2, temp3))
        windowCoords.append(temp)
        windowSize.append(d)

    return windowCoords, windowSize, windowRange_xy


def get_windows(numCells, cellEdgeList, delta, a, b):
    windowCoords = []
    windowSize = []
    windowRange_xy = []

    for i in range(1, numCells):
        # maskImg = mask.get_data()[0, 0, loc, 0, :, :]
        # xmin, xmax, ymin, ymax = np.min(cellEdgeList[inCells[i]][0]), np.max(cellEdgeList[inCells[i]][0]), np.min(
        #     cellEdgeList[inCells[i]][1]), np.max(cellEdgeList[inCells[i]][1])

        xmin, xmax, ymin, ymax = (
            np.min(cellEdgeList[i][1]),
            np.max(cellEdgeList[i][1]),
            np.min(cellEdgeList[i][0]),
            np.max(cellEdgeList[i][0]),
        )

        xmin = xmin - delta if xmin - delta > 0 else 0
        xmax = xmax + delta + 1 if xmax + delta + 1 < a else a
        ymin = ymin - delta if ymin - delta > 0 else 0
        ymax = ymax + delta + 1 if ymax + delta + 1 < b else b
        # tempImg = np.zeros((xmax - xmin, ymax - ymin))
        xy = np.array([xmin, xmax, ymin, ymax])
        windowRange_xy.append(xy)

        x = xmax - xmin
        y = ymax - ymin
        c = np.array([x, y])

        temp1 = cellEdgeList[i][1] - xmin
        temp2 = cellEdgeList[i][0] - ymin

        temp = np.stack((temp1, temp2))
        windowCoords.append(temp)
        windowSize.append(c)

    return windowCoords, windowSize, windowRange_xy


@nb.njit(parallel=True)
def nbget_windows(numCells, cellEdgeList, inCells, delta, a, b):
    windowCoords = nbList()
    windowSize = nbList()
    windowRange_xy = nbList()

    for i in nb.prange(numCells):
        # maskImg = mask.get_data()[0, 0, loc, 0, :, :]
        xmin, xmax, ymin, ymax = (
            np.min(cellEdgeList[i][0]),
            np.max(cellEdgeList[i][0]),
            np.min(cellEdgeList[i][1]),
            np.max(cellEdgeList[i][1]),
        )

        xmin = xmin - delta if xmin - delta > 0 else 0
        xmax = xmax + delta + 1 if xmax + delta + 1 < a else a
        ymin = ymin - delta if ymin - delta > 0 else 0
        ymax = ymax + delta + 1 if ymax + delta + 1 < b else b
        # tempImg = np.zeros((xmax - xmin, ymax - ymin))
        xy = np.array([xmin, xmax, ymin, ymax])
        windowRange_xy.append(xy)

        x = xmax - xmin
        y = ymax - ymin
        c = np.array([x, y])

        temp1 = cellEdgeList[i][0] - xmin
        temp2 = cellEdgeList[i][1] - ymin

        temp = np.stack((temp1, temp2))
        windowCoords.append(temp)
        windowSize.append(c)

        # for j in range(0, len(cellEdgeList[i][0])):
        #     tempImg[test[j], test2[j]] = 1
        # tempImg[test, test2] = 1
        # tempImg[cellEdgeList[i][0] - xmin, cellEdgeList[i][1] - ymin] = 1

        # nbwindows.append(tempImg)
        # maskimgwindow.append(maskImg[xmin:xmax, ymin:ymax])

        # windowspec['xmin'] = xmin
        # windowspec['xmax'] = xmax
        # windowspec['ymin'] = ymin
        # windowspec['ymax'] = ymax

    return windowCoords, windowSize, windowRange_xy


def AdjacencyMatrix2Graph(adjacencyMatrix, cell_center: np.ndarray, cellGraph, name, thr):
    cell_center = pd.DataFrame(cell_center)
    cells = set(cell_center.index)
    fig, ax = plt.subplots(figsize=(17.0, 17.0))
    plt.plot(cell_center.iloc[:, 0], cell_center.iloc[:, 1], ",")
    plt.title("Cell Adjacency Graph, distance <" + str(thr))
    for i, cell_coord in cell_center.iterrows():
        idx = list(cellGraph[i])
        if idx and all(x in cells for x in idx):
            neighbors = cell_center.loc[idx, :]
            lines = []
            for j, neighbor_coord in neighbors.iterrows():
                lines.append([cell_coord, neighbor_coord])

                # dist = adjacencyMatrix[i, j]
                # gap = (neighbor_coord - cell_coord) / 2
                # ax.text(
                #     cell_coord[0] + gap[0],
                #     cell_coord[1] + gap[1],
                #     '%.1f' % dist,
                #     ha='center',
                #     va='center',
                #     fontsize='xx-small'
                # )
            line = mc.LineCollection(lines, colors=[(1, 0, 0, 1)])
            ax.add_collection(line)
    plt.savefig(name, **figure_save_params)
    plt.clf()
    plt.close()


def AdjacencyMatrix2Graph(adjacencyMatrix, cell_center_og: np.ndarray, cellGraph, name, thr):
    cell_center = pd.DataFrame(cell_center_og)

    # check if 3D or 2D - drop z if 3D
    if cell_center.shape[1] == 3:
        cell_center = cell_center.drop(columns=2)

    cells = set(cell_center.index)
    fig, ax = plt.subplots(figsize=(17.0, 17.0))
    plt.plot(cell_center.iloc[:, 0], cell_center.iloc[:, 1], ",")
    plt.title("Cell Adjacency Graph, distance <" + str(thr))
    for i, cell_coord in cell_center.iterrows():
        idx = list(cellGraph[i])
        if idx and all(x in cells for x in idx):
            neighbors = cell_center.loc[idx, :]
            lines = []
            for j, neighbor_coord in neighbors.iterrows():
                lines.append([cell_coord, neighbor_coord])

                # dist = adjacencyMatrix[i, j]
                # gap = (neighbor_coord - cell_coord) / 2
                # ax.text(
                #     cell_coord[0] + gap[0],
                #     cell_coord[1] + gap[1],
                #     '%.1f' % dist,
                #     ha='center',
                #     va='center',
                #     fontsize='xx-small'
                # )
            line = mc.LineCollection(lines, colors=[(1, 0, 0, 1)])
            ax.add_collection(line)
    plt.savefig(name, **figure_save_params)
    plt.clf()
    plt.close()


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


def get_paths(img_dir: Path) -> List[Path]:
    if img_dir.is_dir():
        img_files = []

        for dirpath_str, _, filenames in walk(img_dir):
            dirpath = Path(dirpath_str)
            filenames_usable = set(filenames) - FILENAMES_TO_IGNORE
            img_files.extend(dirpath / filename for filename in filenames_usable)
    else:
        # assume it's a pattern, like Path('some/dir/*.tiff')
        # don't need to filter filenames, because the user passed a
        # glob pattern of exactly what is wanted
        img_files = list(img_dir.parent.glob(img_dir.name))

    return sorted(img_files, key=alphanum_sort_key)


def get_df_format(sub_matrix, s: str, img: IMGstruct, options: Dict) -> (List[str], Any):
    # can use dict switch to speed up access if the formats get > 5
    names = img.get_channel_labels().copy()
    if len(sub_matrix.shape) < 3:
        if "shape" in s:
            header = [i for i in range(1, sub_matrix.shape[1] + 1)]
        elif "PCA" in s:
            names.append("Explained Variance")
            names = list(map(str, names))
            header = names
        elif "Scores" in s:
            header = list(range(2, sub_matrix.shape[1] + 2))
        else:
            header = names
    elif "covar" in s:
        concat = np.reshape(
            sub_matrix, (sub_matrix.shape[0], sub_matrix.shape[1] * sub_matrix.shape[2])
        )
        sub_matrix = concat
        comb = []
        for pair in product(names, names):
            comb.append(":".join(pair))
        header = comb
        options["channel_label_combo"] = header
    else:
        pass

    return header, sub_matrix


def write_2_file(
    sub_matrix, s: str, img: IMGstruct, output_dir: Path, cellidx: list, options: Dict
):
    header, sub_matrix = get_df_format(sub_matrix, s, img, options)
    write_2_csv(header, sub_matrix, s, output_dir, cellidx, options)


def write_2_csv(header: List, sub_matrix, s: str, output_dir: Path, cellidx: list, options: Dict):
    global row_index

    if row_index:
        df = pd.DataFrame(sub_matrix, index=options.get("row_index_names"), columns=header)
    else:
        if len(cellidx) == sub_matrix.shape[0]:
            df = pd.DataFrame(sub_matrix, index=cellidx, columns=header)
        df = pd.DataFrame(
            sub_matrix, index=list(range(1, sub_matrix.shape[0] + 1)), columns=header
        )

    if options.get("debug"):
        print(df)
        print(header)
    f = output_dir / (s + ".csv")

    # column header for indices
    if "PCA" in s:
        df.index.name = "PCA #"
    elif "Scores" in s:
        df.index.name = "Cluster Types"
    else:
        df.index.name = "ID"

    # write out header
    # df.columns = header

    if "Score" in s:
        df.index = options.get("cluster_types")
        df.to_csv(f, header=header, index_label=df.index.name, index=options.get("cluster_types"))
    else:
        df.to_csv(f, header=header, index_label=df.index.name)

    # Sean Donahue - 11/12/20
    key_parts = s.replace("-", "_").split("_")
    key_parts.reverse()
    hdf_key = "/".join(key_parts)
    with pd.HDFStore(output_dir / Path("out.hdf5")) as store:
        store.put(hdf_key, df)


def write_cell_polygs(
    polyg_list: List[np.ndarray],
    pts: np.ndarray,
    cellidx: list,
    filename: str,
    output_dir: Path,
    options: Dict,
):
    coord_pairs = []
    for i in range(0, len(polyg_list)):
        tlist = str(
            [[round(i, 4), round(j, 4)] for i, j in zip(polyg_list[i][:, 0], polyg_list[i][:, 1])]
        )
        coord_pairs.append(tlist)

    coord_pairs = np.asarray(coord_pairs)

    write_2_csv(
        ["Shape"],
        coord_pairs,
        filename + "-cell_polygons_spatial",
        output_dir,
        cellidx,
        options,
    )
    # df = pd.DataFrame({0: coord_pairs}, index=cellidx)
    # if options.get("debug"):
    #     print(df)
    # f = output_dir / (filename + "-cell_polygons_spatial.csv")
    # df.index.name = "ID"
    # df.to_csv(f, header=["Shape"])

    # write out pts - 100 outlines
    header = ["Scaling Factor"]
    for i in range(1, 201):
        header.append(i)
    write_2_csv(header, pts, filename + "-normalized_cell_outlines", output_dir, cellidx, options)


def build_matrix(
    im: IMGstruct,
    mask: MaskStruct,
    masked_imgs_coord: List[np.ndarray],
    j: int,
    omatrix: np.ndarray,
) -> np.ndarray:
    if j == 0:
        return np.zeros(
            (
                im.get_data().shape[1],
                mask.get_data().shape[2],
                len(masked_imgs_coord),
                im.get_data().shape[2],
                im.get_data().shape[2],
            )
        )
    else:
        return omatrix


def build_vector(
    im: IMGstruct,
    mask: MaskStruct,
    masked_imgs_coord: List[np.ndarray],
    j: int,
    omatrix: np.ndarray,
) -> np.ndarray:
    if j == 0:
        return np.zeros(
            (
                im.get_data().shape[1],
                mask.get_data().shape[2],
                len(masked_imgs_coord),
                im.get_data().shape[2],
                1,
            )
        )
    else:
        return omatrix


def clusterchannels(
    im: IMGstruct, fname: str, output_dir: Path, inCells: list, options: Dict
) -> np.ndarray:
    """
    cluster all channels using PCA
    """
    print("Dimensionality Reduction of image channels...")
    if options.get("debug"):
        print("Image dimensions before reduction: ", im.get_data().shape)
    pca_channels = PCA(n_components=options.get("num_channelPCA_components"))
    channvals = im.get_data()[0, 0, :, :, :, :]
    keepshape = channvals.shape
    channvals = channvals.reshape(
        channvals.shape[0], channvals.shape[1] * channvals.shape[2] * channvals.shape[3]
    )
    channvals = channvals.transpose()
    if options.get("debug"):
        print(channvals.shape)

    if options.get("zscore_norm"):
        channvals = stats.zscore(channvals)

    channvals_full = channvals.copy()
    tries = 0
    while True:
        try:
            m = pca_channels.fit(channvals)
            break
        except Exception as e:
            print(e)
            print("Exceptions caught: ", tries)
            if tries == 0:
                pca_channels = PCA(
                    n_components=options.get("num_channelPCA_components"), svd_solver="randomized"
                )
                tries += 1
            else:
                print("halving the dataset...")
                n_samples = int(channvals.shape[0] / 2)
                idx = np.random.choice(channvals_full.shape[0], n_samples, replace=False)
                channvals = channvals_full[idx, :]
                # keepshape[1] = int(keepshape[1]/2) + 1
                # keepshape[2] = keepshape[2]/2
                # keepshape[3] = keepshape[3]/2
                # pca_channels.fit(reduced_channvals)
                # reducedim = pca_channels.transform(channvals)

    reducedim = m.transform(channvals_full)
    if options.get("debug"):
        print("PCA Channels:", pca_channels, sep="\n")
        print("Explained variance ratio: ", pca_channels.explained_variance_ratio_)
        print("Singular values: ", pca_channels.singular_values_)
        print("Image dimensions before transposing & reshaping: ", reducedim.shape)
    reducedim = reducedim.transpose()
    reducedim = reducedim.reshape(reducedim.shape[0], keepshape[1], keepshape[2], keepshape[3])
    if options.get("debug"):
        print("Image dimensions after transposing and reshaping: ", reducedim.shape)

    # find pca comp and explained variance and concatenate
    a = pca_channels.explained_variance_ratio_
    b = abs(pca_channels.components_)
    c = np.column_stack((b, a))

    write_2_file(c, fname + "-channelPCA_summary", im, output_dir, inCells, options)

    return reducedim


def plotprincomp(
    reducedim: np.ndarray, bestz: list, filename: str, output_dir: Path, options: Dict
) -> np.ndarray:
    print("Plotting PCA image...")
    if options.get("image_dimension") == "2D":
        bestz = bestz[0]
        tp = (1, 2, 0)
    else:
        tp = (1, 2, 3, 0)

    reducedim = reducedim[:, bestz, :, :]
    # k = reducedim.shape[0]
    # if k > 2:
    #     reducedim = reducedim[0:3, :, :]
    # else:
    #     rzeros = np.zeros(reducedim.shape)
    #     reducedim[k:3, :, :] = rzeros[k:3, :, :]

    # plotim = reducedim.transpose(1, 2, 0)
    plotim = np.transpose(reducedim, tp)

    if options.get("debug"):
        print("Before Transpose:", reducedim.shape)
        print("After Transpose: ", plotim.shape)

    for i in range(0, 3):
        cmin = plotim[..., i].min()
        cmax = plotim[..., i].max()
        if options.get("debug"):
            print("Min and Max before normalization: ", cmin, cmax)
        plotim[..., i] = ((plotim[..., i] - cmin) / (cmax - cmin)) * 255.0
        cmin = plotim[..., i].min()
        cmax = plotim[..., i].max()
        if options.get("debug"):
            print("Min and Max after normalization: ", cmin, cmax)

    plotim = plotim.round().astype(np.uint8)

    # save img
    # save_image(plotim, output_dir / filename, options)
    if plotim.ndim <= 3:
        img = Image.fromarray(plotim, mode="RGB")
        img.save(output_dir / filename)
    else:
        apng = APNG()
        # forward
        for j in range(len(plotim)):
            in_mem_file = io.BytesIO()
            i = Image.fromarray(plotim[j], mode="RGB")
            i.save(in_mem_file, format="PNG")
            in_mem_file.seek(0)
            img_bytes = in_mem_file.read()
            png = PNG.from_bytes(img_bytes)
            apng.append(png, delay=options.get("apng_delay"))

        # backward
        for j in range(len(plotim)):
            in_mem_file = io.BytesIO()
            i = Image.fromarray(plotim[-j], mode="RGB")
            i.save(in_mem_file, format="PNG")
            in_mem_file.seek(0)
            img_bytes = in_mem_file.read()
            png = PNG.from_bytes(img_bytes)
            apng.append(png, delay=options.get("apng_delay"))

        apng.save(output_dir / filename)

    return plotim


def SNR(im: IMGstruct, filename: str, output_dir: Path, inCells: list, options: Dict):
    """
    signal to noise ratio of each channel of the image w/ z score and otsu thresholding
    """

    global row_index

    print("Calculating Signal to Noise Ratio in image...")
    channvals = im.get_data()[0, 0, :, :, :, :]
    zlist = []
    channelist = []

    channvals_z = channvals.reshape(
        channvals.shape[0], channvals.shape[1] * channvals.shape[2] * channvals.shape[3]
    )
    channvals_z = channvals_z.transpose()

    m = channvals_z.mean(0)
    sd = channvals_z.std(axis=0, ddof=0)
    # snr_channels = np.where(sd == 0, 0, m / sd)
    snr_channels = m / sd
    # snr_channels = snr_channels[np.newaxis, :]
    # snr_channels = snr_channels.tolist()

    # define otsu threshold
    for i in range(0, channvals.shape[0]):
        img_2D = channvals[i, :, :, :].astype("int")
        img_2D = img_2D.reshape((img_2D.shape[0] * img_2D.shape[1], img_2D.shape[2]))

        # try:
        # nbins implement
        max_img = np.max(img_2D)
        max_img_dtype = np.iinfo(img_2D.dtype).max
        factor = np.floor(max_img_dtype / max_img)
        img_2D_factor = img_2D * factor

        thresh = threshold_otsu(img_2D_factor)
        thresh = max(thresh, factor)

        above_th = img_2D_factor > thresh
        below_th = img_2D_factor <= thresh
        # if thresh == 0:
        #     print('Otsu threshold returned 0')
        #     fbr = 'N/A'
        # else:
        mean_a = np.mean(img_2D_factor[above_th])
        mean_b = np.mean(img_2D_factor[below_th])
        # take ratio above thresh / below thresh
        fbr = mean_a / mean_b
        # except ValueError:
        #     print('Value Error encountered')
        #     fbr = 0

        channelist.append(fbr)

    channelist = np.asarray(channelist)

    snrs = np.stack([snr_channels, channelist])

    # check for nans
    snrs[np.isnan(snrs)] = 0

    # add index identifier
    row_index = 1
    options["row_index_names"] = ["Z-Score", "Otsu"]

    # write out 2 rows
    write_2_csv(im.get_channel_labels(), snrs, filename + "-SNR", output_dir, inCells, options)

    # turn index boolean off
    row_index = 0

    # return snr_channels, channelist


def voxel_cluster(im: IMGstruct, options: Dict) -> np.ndarray:
    """
    cluster multichannel image into superpixels
    """
    print("Clustering voxels into superpixels...")
    if im.get_data().shape[0] > 1:
        raise NotImplementedError("images with > 1 time point are not supported yet")

    channvals = im.get_data()[0, 0, :, :, :, :]
    keepshape = channvals.shape
    channvals = channvals.reshape(
        channvals.shape[0], channvals.shape[1] * channvals.shape[2] * channvals.shape[3]
    )
    channvals = channvals.transpose()
    # for some reason, voxel values are occasionally NaNs (especially edge rows)
    channvals[np.where(np.isnan(channvals))] = 0

    if options.get("zscore_norm"):
        channvals = stats.zscore(channvals)

    if options.get("debug"):
        print("Multichannel dimensions: ", channvals.shape)
    # get random sampling of pixels in 2d array
    np.random.seed(0)
    sampling = float(options.get("precluster_sampling"))
    samples = math.ceil(sampling * channvals.shape[0])
    # lower bound threshold on pixels
    if samples < options.get("precluster_threshold"):
        samples = options.get("precluster_threshold")
    idx = np.random.choice(channvals.shape[0], samples)
    channvals_random = channvals[idx]
    # kmeans clustering of random sampling
    print("Clustering random sample of voxels...")
    stime = time.monotonic() if options.get("debug") else None

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

    voxelbycluster = KMeans(n_clusters=options.get("num_voxelclusters"), random_state=0).fit(
        channvals_random
    )

    if options.get("debug"):
        print("random sample voxel cluster runtime: " + str(time.monotonic() - stime))
    cluster_centers = voxelbycluster.cluster_centers_
    # fast kmeans clustering with inital centers
    print("Clustering voxels with initialized centers...")
    stime = time.monotonic() if options.get("debug") else None
    voxelbycluster = KMeans(
        n_clusters=options.get("num_voxelclusters"),
        init=cluster_centers,
        random_state=0,
        max_iter=100,
        verbose=0,
        n_init=1,
    ).fit(channvals)
    # voxelbycluster = KMeans(n_clusters=options.get("num_voxelclusters"), random_state=0).fit(channvals)
    if options.get("debug"):
        print("Voxel cluster runtime: ", time.monotonic() - stime)
    # returns a vector of len number of voxels and the vals are the cluster numbers
    voxelbycluster_labels = voxelbycluster.labels_
    voxelbycluster_labels = voxelbycluster_labels.reshape(keepshape[1], keepshape[2], keepshape[3])

    if options.get("debug"):
        print("Cluster Label dimensions: ", voxelbycluster_labels.shape)
        print("Number of unique labels:")
        print(len(np.unique(voxelbycluster_labels)))

    # add 1 to cluster so index starts @1
    voxelbycluster_labels += 1

    return voxelbycluster_labels


def findmarkers(clustercenters: np.ndarray, options: Dict) -> List:
    """
    find a set of markers that have the largest variance without correlations among them
    """
    clustercenters = np.transpose(clustercenters)
    markerlist = []
    markergoal = options.get("num_markers")
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
        markerlist = list(np.argsort(-clustercenters.ravel()))
        # remap back to original feature length
        markerlist = [np.unravel_index(i, clustercenters.shape)[0] for i in markerlist]
        markerlist_markergoal = markerlist[:markergoal]
        # check to see if index is identical or not
        if len(markerlist_markergoal) != len(set(markerlist_markergoal)):
            markerlist_markergoal = set()
            idx = 0
            while len(set(markerlist_markergoal)) < markergoal:
                markerlist_markergoal.add(markerlist[idx])
                idx += 1

        return list(markerlist_markergoal)

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


def matchNShow_markers(
    clustercenters: np.ndarray, markerlist: List, features: List[str], options: Dict
) -> (Any, List[str]):
    """
    get the markers to indicate what the respective clusters represent
    """

    markers = [features[i] for i in markerlist]
    table = clustercenters[:, markerlist]

    if options.get("debug"):
        print(markerlist)
        print(table)

    return table, markers


def write_ometiff(im: IMGstruct, output_dir: Path, options: Dict, *argv):
    print("Writing out ometiffs for visualizations...")
    pcaimg = argv[0]
    # pcaimg = pcaimg.astype(np.int32)

    superpixel = argv[1]
    # superpixel = get_last2d(argv[1], bestz[0])
    # superpixel = superpixel.astype(np.int32)
    # superpixel = superpixel[np.newaxis, :, :]

    if options.get("image_dimension") == "3D":
        pca_dim_order = "CZYX"
        # superpixel_dim_order = 'ZYX'
        pca_tp = (3, 0, 1, 2)
    else:
        pca_dim_order = "CYX"
        # superpixel_dim_order = 'YX'
        pca_tp = (2, 0, 1)

    superpixel_dim_order = "ZYX"
    pcaimg = np.transpose(pcaimg, pca_tp)

    s = ["-channel_pca.ome.tiff", "-superpixel.ome.tiff"]
    f = [output_dir / (im.get_name() + s[0]), output_dir / (im.get_name() + s[1])]

    check_file_exist(f)
    writer = OmeTiffWriter()
    writer.save(pcaimg, f[0], image_name=im.get_name(), dim_order=pca_dim_order)
    writer = OmeTiffWriter()
    writer.save(superpixel, f[1], image_name=im.get_name(), dim_order=superpixel_dim_order)


def check_file_exist(paths: Path):
    for i in paths:
        if i.is_file():
            i.unlink()
        else:
            continue


def cell_cluster_IDs(
    filename: str,
    output_dir: Path,
    i: int,
    maskchs: list,
    inCells: list,
    options: Dict,
    celltype_labels: Path,
    *argv,
):
    allClusters = argv[0]
    for idx in range(1, len(argv)):
        allClusters = np.column_stack((allClusters, argv[idx]))

    # init
    ignore_col = []

    # hard coded --> find a way to automate the naming
    if not options.get("skip_outlinePCA"):
        # write_2_csv(
        #     list(
        #         [
        #             "K-Means [Texture]",
        #             "K-Means [Mean] Expression",
        #             "K-Means [Covariance] Expression",
        #             "K-Means [Total] Expression",
        #             "K-Means [Mean-All-SubRegions] Expression",
        #             "K-Means [Shape-Vectors]",
        #             "K-Means [tSNE_All_Features]",
        #             "K-Means [Shape-Vectors Normalized]",
        #             "K-Means [UMAP_All_Features]",
        #         ]
        #     ),
        #     allClusters,
        #     filename + "-" + maskchs[i] + "_cluster",
        #     output_dir,
        #     inCells,
        #     options,
        # )
        column_names = [
            "K-Means [Texture]",
            "K-Means [Mean] Expression",
            "K-Means [Covariance] Expression",
            "K-Means [Total] Expression",
            "K-Means [Mean-All-SubRegions] Expression",
            "K-Means [Shape-Vectors]",
            "K-Means [tSNE_All_Features]",
            "K-Means [Shape-Vectors Normalized]",
            "K-Means [UMAP_All_Features]",
        ]
    else:
        # write_2_csv(
        #     list(
        #         [
        #             "K-Means [Texture]",
        #             "K-Means [Mean] Expression",
        #             "K-Means [Covariance] Expression",
        #             "K-Means [Total] Expression",
        #             "K-Means [Mean-All-SubRegions] Expression",
        #             "K-Means [tSNE_All_Features]",
        #             "K-Means [UMAP_All_Features]",
        #         ]
        #     ),
        #     allClusters,
        #     filename + "-" + maskchs[i] + "_cluster",
        #     output_dir,
        #     inCells,
        #     options,
        # )
        column_names = [
            "K-Means [Texture]",
            "K-Means [Mean] Expression",
            "K-Means [Covariance] Expression",
            "K-Means [Total] Expression",
            "K-Means [Mean-All-SubRegions] Expression",
            "K-Means [tSNE_All_Features]",
            "K-Means [UMAP_All_Features]",
        ]

    # convert to a dataframe
    new_allClusters = pd.DataFrame(data=allClusters, index=inCells, columns=column_names)
    # index starting by 1
    new_allClusters += 1

    # read in celltype labels if exists
    if celltype_labels:
        celltype_labels = pd.read_csv(celltype_labels)
        new_allClusters = pd.merge(
            new_allClusters, celltype_labels, left_index=True, right_on="ID"
        )
        new_allClusters.index = inCells  # index starting at 1
        new_allClusters = new_allClusters.drop(columns="ID")
        ignore_col.append(celltype_labels.columns[1])

    if options.get("skip_texture"):
        ignore_col.append("K-Means [Texture]")

    new_allClusters = transform_df(new_allClusters, ignore_column=ignore_col)
    column_names = new_allClusters.columns.to_list()
    new_allClusters_np = new_allClusters.to_numpy()

    # convert back to numpy array to be written out
    write_2_csv(
        column_names,
        new_allClusters_np,
        filename + "-" + maskchs[i] + "_cluster",
        output_dir,
        inCells,
        options,
    )

    return new_allClusters


def transform_df(df, ignore_column=None):
    if ignore_column != None:
        ignore_column_df = df.loc[:, ignore_column]
        df = df.drop(columns=ignore_column)

    clusters = np.unique(df.values)
    drop_rows = []
    df_copy = df.copy()

    for i in clusters:
        # Count all unique values across all columns
        values_counter = df_copy.apply(lambda x: x.value_counts()).fillna(0).astype("int")

        row, column = find_max_value_drop_rows(values_counter, drop_rows)
        indices = df_copy[df_copy[column] == row].index.tolist()
        match_counter = (
            df_copy.apply(lambda x: x.loc[indices].value_counts())
            .drop(drop_rows, errors="ignore")
            .idxmax()
            .dropna()
            .astype("int")
        )

        if (match_counter.values == i).all():
            drop_rows.append(i)
            continue

        for col in match_counter.index:
            if i != match_counter[col]:
                mapping = {i: match_counter[col], match_counter[col]: i}
                df_copy.loc[:, col] = df_copy.loc[:, col].map(lambda x: mapping.get(x, x))

        drop_rows.append(i)

    if ignore_column != None:
        # merge the dropped column back to the new one
        df_copy = pd.concat([ignore_column_df, df_copy], axis=1)

    return df_copy


def find_max_value_drop_rows(df, drop_rows):
    # Drop rows in the list
    df = df.drop(drop_rows, errors="ignore")

    # Find the location of the largest value
    max_row = df.stack().idxmax()

    # Extract row index and column name
    max_row_index = max_row[0]
    max_column_name = max_row[1]

    return max_row_index, max_column_name


def plot_img(cluster_im: np.ndarray, bestz: list, filename: str, output_dir: Path, options: dict):
    cluster_im = get_last2d(cluster_im, bestz, options)

    save_image(cluster_im, output_dir / filename, options)


def plot_imgs(
    filename: str,
    output_dir: Path,
    i: int,
    maskchs: List,
    options: Dict,
    clusters_df,
    cluster_img_list: List,
):
    # find the max cluster
    if "DeepCell Type" in clusters_df.columns:
        options["max_cluster"] = clusters_df.iloc[:, 1:].values.max()
    else:
        options["max_cluster"] = clusters_df.values.max()

    # UPDATE: NEED TO AUTOMATE AND NOT HARDCODE

    plot_img(
        cluster_img_list[1],
        [0],
        filename + "-clusterbyMeansper" + maskchs[i] + ".png",
        output_dir,
        options,
    )
    plot_img(
        cluster_img_list[2],
        [0],
        filename + "-clusterbyCovarper" + maskchs[i] + ".png",
        output_dir,
        options,
    )
    plot_img(
        cluster_img_list[3],
        [0],
        filename + "-clusterbyTotalper" + maskchs[i] + ".png",
        output_dir,
        options,
    )

    if i == 0:
        if not options.get("skip_outlinePCA"):
            plot_img(
                cluster_img_list[5], [0], filename + "-Cluster_Shape.png", output_dir, options
            )
            plot_img(
                cluster_img_list[7],
                [0],
                filename + "-Cluster_ShapeNormalized.png",
                output_dir,
                options,
            )
            plot_img(
                cluster_img_list[8], [0], filename + "-clusterbyUMAP.png", output_dir, options
            )

            plot_img(
                cluster_img_list[0], [0], filename + "-clusterbyTexture.png", output_dir, options
            )
            plot_img(
                cluster_img_list[4], [0], filename + "-clusterbyMeansAll.png", output_dir, options
            )
            plot_img(
                cluster_img_list[6],
                [0],
                filename + "-clusterbytSNEAllFeatures.png",
                output_dir,
                options,
            )
        else:
            plot_img(
                cluster_img_list[0], [0], filename + "-clusterbyTexture.png", output_dir, options
            )
            plot_img(
                cluster_img_list[4], [0], filename + "-clusterbyMeansAll.png", output_dir, options
            )
            plot_img(
                cluster_img_list[5],
                [0],
                filename + "-clusterbytSNEAllFeatures.png",
                output_dir,
                options,
            )
            plot_img(
                cluster_img_list[6], [0], filename + "-clusterbyUMAP.png", output_dir, options
            )

    # remove options max cluster
    options.pop("max_cluster")


def make_legends(
    feature_names,
    feature_covar,
    feature_meanall,
    filename: str,
    output_dir: Path,
    i: int,
    maskchn: List,
    inCells: list,
    options: Dict,
    *argv,
):
    # make legend once
    if i == 0:
        print("Finding mean ALL cluster markers...")
        retmarkers = findmarkers(argv[3], options)
        table, markers = matchNShow_markers(argv[3], retmarkers, feature_meanall, options)
        write_2_csv(
            markers, table, filename + "-cluster_meanALLCH_legend", output_dir, inCells, options
        )
        showlegend(markers, table, filename + "-cluster_meanALLCH_legend.pdf", output_dir)

        if not options.get("skip_outlinePCA"):
            feature_shape = ["shapefeat " + str(ff) for ff in range(0, argv[4].shape[1])]
            print("Finding cell shape cluster markers...")
            retmarkers = findmarkers(argv[4], options)
            table, markers = matchNShow_markers(argv[4], retmarkers, feature_shape, options)
            write_2_csv(
                markers,
                table,
                filename + "-clustercell_cellshape_legend",
                output_dir,
                inCells,
                options,
            )
            showlegend(markers, table, filename + "-clustercells_cellshape_legend.pdf", output_dir)

            # normalized cell shape
            norm_feature_shape = ["shapefeat " + str(ff) for ff in range(0, argv[4].shape[1])]
            print("Finding cell shape normalized cluster markers")
            retmarkers = findmarkers(argv[7], options)
            table, markers = matchNShow_markers(argv[7], retmarkers, norm_feature_shape, options)
            write_2_csv(
                markers,
                table,
                filename + "-clustercell_cellshapenormalized_legend",
                output_dir,
                inCells,
                options,
            )
            showlegend(
                markers,
                table,
                filename + "-clustercells_cellshapenormalized_legend.pdf",
                output_dir,
            )

            feature_shape = ["shapefeat " + str(ff) for ff in range(0, argv[4].shape[1])]
            print("Finding cell shape cluster markers...")
            retmarkers = findmarkers(argv[4], options)
            table, markers = matchNShow_markers(argv[4], retmarkers, feature_shape, options)
            write_2_csv(
                markers,
                table,
                filename + "-clustercell_cellshape_legend",
                output_dir,
                inCells,
                options,
            )
            showlegend(markers, table, filename + "-clustercells_cellshape_legend.pdf", output_dir)

            print("Finding cell texture cluster markers...")
            retmarkers = findmarkers(argv[5][0], options)
            table, markers = matchNShow_markers(argv[5][0], retmarkers, argv[5][1], options)
            write_2_csv(
                markers,
                table,
                filename + "-clustercell_texture_legend",
                output_dir,
                inCells,
                options,
            )
            showlegend(markers, table, filename + "-clustercells_texture_legend.pdf", output_dir)

            print("Finding cell tsne all features cluster markers...")
            retmarkers = findmarkers(argv[6][0], options)
            table, markers = matchNShow_markers(argv[6][0], retmarkers, argv[6][1], options)
            write_2_csv(
                markers,
                table,
                filename + "-clustercell_tSNE_legend",
                output_dir,
                inCells,
                options,
            )
            showlegend(markers, table, filename + "-clustercells_texture_legend.pdf", output_dir)
        else:
            print("Finding cell texture cluster markers...")
            retmarkers = findmarkers(argv[4][0], options)
            table, markers = matchNShow_markers(argv[4][0], retmarkers, argv[4][1], options)
            write_2_csv(
                markers,
                table,
                filename + "-clustercell_texture_legend",
                output_dir,
                inCells,
                options,
            )
            showlegend(markers, table, filename + "-clustercells_texture_legend.pdf", output_dir)

            print("Finding cell tsne all features cluster markers...")
            retmarkers = findmarkers(argv[5][0], options)
            table, markers = matchNShow_markers(argv[5][0], retmarkers, argv[5][1], options)
            write_2_csv(
                markers,
                table,
                filename + "-clustercell_tSNE_legend",
                output_dir,
                inCells,
                options,
            )
            showlegend(markers, table, filename + "-clustercells_texture_legend.pdf", output_dir)

    print("Legend for mask channel: " + str(i))

    for j in range(len(argv)):
        # hard coded for argv idx and - psuedo switch -- might be a more efficient way
        if j == 0:
            print("Finding mean cluster markers...")
            retmarkers = findmarkers(argv[j], options)
            table, markers = matchNShow_markers(argv[j], retmarkers, feature_names, options)
            write_2_csv(
                markers,
                table,
                filename + "-cluster" + maskchn[i] + "_mean_legend",
                output_dir,
                inCells,
                options,
            )
            showlegend(
                markers, table, filename + "-cluster" + maskchn[i] + "_mean_legend.pdf", output_dir
            )

        elif j == 1:
            print("Finding covariance cluster markers...")
            retmarkers = findmarkers(argv[j], options)
            table, markers = matchNShow_markers(argv[j], retmarkers, feature_covar, options)
            write_2_csv(
                markers,
                table,
                filename + "-cluster" + maskchn[i] + "_covariance_legend",
                output_dir,
                inCells,
                options,
            )
            showlegend(
                markers,
                table,
                filename + "-cluster" + maskchn[i] + "_covariance_legend.pdf",
                output_dir,
            )

        elif j == 2:
            print("Finding total cluster markers...")
            retmarkers = findmarkers(argv[j], options)
            table, markers = matchNShow_markers(argv[j], retmarkers, feature_names, options)
            write_2_csv(
                markers,
                table,
                filename + "-cluster" + maskchn[i] + "_total_legend",
                output_dir,
                inCells,
                options,
            )
            showlegend(
                markers,
                table,
                filename + "-cluster" + maskchn[i] + "_total_legend.pdf",
                output_dir,
            )


def save_all(
    filename: str,
    im: IMGstruct,
    mask: MaskStruct,
    output_dir: Path,
    cellidx: list,
    options: Dict,
    *argv,
):
    # hard coded for now
    print("Writing to csv all matrices...")
    mean_vector = argv[0]
    covar_matrix = argv[1]
    total_vector = argv[2]

    if not options.get("skip_outlinePCA"):
        outline_vectors = argv[3]
        write_2_file(outline_vectors, filename + "-cell_shape", im, output_dir, cellidx, options)

        # normalize shape vectors
        norm_shape_vectors = argv[4]
        write_2_file(
            norm_shape_vectors,
            filename + "-cell_shape_normalized",
            im,
            output_dir,
            cellidx,
            options,
        )

    write_2_file(
        mean_vector[0, -1, :, :, 0],
        filename + "-cell_channel_meanAll",
        im,
        output_dir,
        cellidx,
        options,
    )
    # write_2_file(texture_v[0, -1, :, :, 0], filename + '-cell_channel_textures', im, output_dir, options)

    for i in range(len(mask.channel_labels)):
        write_2_file(
            mean_vector[0, i, :, :, 0],
            filename + "-" + mask.get_channel_labels()[i] + "_channel_mean",
            im,
            output_dir,
            cellidx,
            options,
        )
        write_2_file(
            covar_matrix[0, i, :, :, :],
            filename + "-" + mask.get_channel_labels()[i] + "_channel_covar",
            im,
            output_dir,
            cellidx,
            options,
        )
        write_2_file(
            total_vector[0, i, :, :, 0],
            filename + "-" + mask.get_channel_labels()[i] + "_channel_total",
            im,
            output_dir,
            cellidx,
            options,
        )


def cell_analysis(
    im: IMGstruct,
    mask: MaskStruct,
    filename: str,
    bestz: int,
    output_dir: Path,
    seg_n: int,
    cellidx: list,
    options: Dict,
    celltype_labels: Path,
    *argv,
):
    """
    cluster and statisical analysis done on cell:
    clusters/maps and makes a legend out of the most promient channels and writes them to a csv
    """
    # cellidx = mask.get_cell_index()
    stime = time.monotonic() if options.get("debug") else None
    # hard coded for now
    mean_vector = argv[0]
    covar_matrix = argv[1]
    total_vector = argv[2]

    if not options.get("skip_outlinePCA"):
        shape_vectors = argv[3]
        texture_vectors = argv[4][0]
        texture_channels = argv[4][1]
        norm_shape_vectors = argv[5]
    else:
        texture_vectors = argv[3][0]
        texture_channels = argv[3][1]

    # get channel labels
    maskchs = mask.get_channel_labels()
    feature_names = im.get_channel_labels()
    feature_covar = options.get("channel_label_combo")

    # make unique channel names
    d = int(len(feature_names) * 4 / len(maskchs))
    new_list = [entry for entry in maskchs for _ in range(d)]
    add_list = ["-"] * len(new_list)
    feature_meanall = [a + b for a, b in zip(new_list, add_list)]
    feature_meanall = [a + b for a, b in zip(feature_meanall, feature_names * d)]
    # feature_meanall = feature_names + feature_names + feature_names + feature_names

    # all clusters List with scores
    all_clusters = []
    types_list = []

    # features only clustered once
    meanAll_vector_f = cell_cluster_format(mean_vector, -1, options)
    clustercells_uvall, clustercells_uvallcenters = cell_cluster(
        meanAll_vector_f, types_list, all_clusters, "mean-all", options, output_dir, cellidx
    )  # -1 means use all segmentations

    if options.get("skip_texture"):
        options["texture_flag"] = True

    texture_matrix = cell_cluster_format(texture_vectors, -1, options)
    clustercells_texture, clustercells_texturecenters = cell_cluster(
        texture_matrix, types_list, all_clusters, "texture", options, output_dir, cellidx
    )

    # mean all map
    # cluster_cell_imguall = cell_map(
    #     mask, clustercells_uvall, seg_n, options
    # )  # 0=use first segmentation to map

    #
    # cluster_cell_texture = cell_map(mask, clustercells_texture, seg_n, options)

    if not options.get("skip_outlinePCA"):
        clustercells_shapevectors, shapeclcenters = shape_cluster(
            shape_vectors, types_list, all_clusters, "shape-unnorm", options, output_dir, cellidx
        )
        # clustercells_shape = cell_map(mask, clustercells_shapevectors, seg_n, options)

        clustercells_norm_shapevectors, normshapeclcenters = shape_cluster(
            norm_shape_vectors,
            types_list,
            all_clusters,
            "shape-norm",
            options,
            output_dir,
            cellidx,
        )
        # clustercells_norm_shape = cell_map(mask, clustercells_norm_shapevectors, seg_n, options)

    if options.get("skip_outlinePCA"):
        (
            clustercells_tsneAll,
            clustercells_tsneAllcenters,
            tsneAll_header,
            clustercells_umapAll,
            clustercells_umapAllcenters,
        ) = DR_AllFeatures(
            all_clusters,
            types_list,
            filename,
            cellidx,
            output_dir,
            options,
            covar_matrix,
            total_vector,
            meanAll_vector_f,
            texture_matrix,
        )
    else:
        (
            clustercells_tsneAll,
            clustercells_tsneAllcenters,
            tsneAll_header,
            clustercells_umapAll,
            clustercells_umapAllcenters,
        ) = DR_AllFeatures(
            all_clusters,
            types_list,
            filename,
            cellidx,
            output_dir,
            options,
            covar_matrix,
            total_vector,
            meanAll_vector_f,
            norm_shape_vectors,
            texture_matrix,
        )

    # tsne
    # cluster_cell_imgtsneAll = cell_map(mask, clustercells_tsneAll, seg_n, options)
    # umap
    # cluster_cell_imgumapAll = cell_map(mask, clustercells_umapAll, seg_n, options)

    # for each channel in the mask
    for i in range(len(maskchs)):
        # for i in range(1):
        seg_n = mask.get_labels(maskchs[i])

        # format the feature arrays accordingly
        mean_vector_f = cell_cluster_format(mean_vector, seg_n, options)
        covar_matrix_f = cell_cluster_format(covar_matrix, seg_n, options)
        total_vector_f = cell_cluster_format(total_vector, seg_n, options)

        # cluster by mean and covar using just cell segmentation mask
        print("Clustering cells and getting back labels and centers...")
        clustercells_uv, clustercells_uvcenters = cell_cluster(
            mean_vector_f,
            types_list,
            all_clusters,
            "mean-" + maskchs[i],
            options,
            output_dir,
            cellidx,
        )
        clustercells_cov, clustercells_covcenters = cell_cluster(
            covar_matrix_f,
            types_list,
            all_clusters,
            "covar-" + maskchs[i],
            options,
            output_dir,
            cellidx,
        )
        clustercells_total, clustercells_totalcenters = cell_cluster(
            total_vector_f,
            types_list,
            all_clusters,
            "total-" + maskchs[i],
            options,
            output_dir,
            cellidx,
        )

        # map back to the mask segmentation of indexed cell region
        # print("Mapping cell index in segmented mask to cluster IDs...")
        # cluster_cell_imgu = cell_map(mask, clustercells_uv, seg_n, options)
        # cluster_cell_imgcov = cell_map(mask, clustercells_cov, seg_n, options)
        # cluster_cell_imgtotal = cell_map(mask, clustercells_total, seg_n, options)
        # cluster_cell_imgtsneAll = cell_map(mask, clustercells_tsneAll, seg_n, options)
        print("Getting markers that separate clusters to make legend...")
        if not options.get("skip_outlinePCA"):
            # get markers for each respective cluster & then save the legend/markers
            make_legends(
                feature_names,
                feature_covar,
                feature_meanall,
                filename,
                output_dir,
                i,
                maskchs,
                cellidx,
                options,
                clustercells_uvcenters,
                clustercells_covcenters,
                clustercells_totalcenters,
                clustercells_uvallcenters,
                shapeclcenters,
                [clustercells_texturecenters, texture_channels],
                [clustercells_tsneAllcenters, tsneAll_header],
                normshapeclcenters,
                [clustercells_umapAllcenters, tsneAll_header],
            )
            # save all clusterings to one csv
            print("Writing out all cell cluster IDs for all cell clusterings...")
            clusterids_df = cell_cluster_IDs(
                filename,
                output_dir,
                i,
                maskchs,
                cellidx,
                options,
                celltype_labels,
                clustercells_texture,
                clustercells_uv,
                clustercells_cov,
                clustercells_total,
                clustercells_uvall,
                clustercells_shapevectors,
                clustercells_tsneAll,
                clustercells_norm_shapevectors,
                clustercells_umapAll,
            )

            # get images of cluster ids mapped to mask
            list_of_cluster_imgs = cell_map(mask, clusterids_df, seg_n, options)

            # plots the cluster imgs for the best z plane
            print("Saving pngs of cluster plots by best focal plane...")
            # plot_imgs(
            #     filename,
            #     output_dir,
            #     i,
            #     maskchs,
            #     options,
            #     cluster_cell_imgu[bestz],
            #     cluster_cell_imgcov[bestz],
            #     cluster_cell_imguall[bestz],
            #     cluster_cell_imgtotal[bestz],
            #     cluster_cell_texture[bestz],
            #     cluster_cell_imgtsneAll[bestz],
            #     clustercells_shape[bestz],
            #     clustercells_norm_shape[bestz],
            #     cluster_cell_imgumapAll[bestz],
            # )
            plot_imgs(
                filename,
                output_dir,
                i,
                maskchs,
                options,
                clusterids_df,
                list_of_cluster_imgs,
            )
        else:
            make_legends(
                feature_names,
                feature_covar,
                feature_meanall,
                filename,
                output_dir,
                i,
                maskchs,
                cellidx,
                options,
                clustercells_uvcenters,
                clustercells_covcenters,
                clustercells_totalcenters,
                clustercells_uvallcenters,
                [clustercells_texturecenters, texture_channels],
                [clustercells_tsneAllcenters, tsneAll_header],
                [clustercells_umapAllcenters, tsneAll_header],
            )
            # save all clusterings to one csv
            print("Writing out all cell cluster IDs for all cell clusterings...")
            clusterids_df = cell_cluster_IDs(
                filename,
                output_dir,
                i,
                maskchs,
                cellidx,
                options,
                celltype_labels,
                clustercells_texture,
                clustercells_uv,
                clustercells_cov,
                clustercells_total,
                clustercells_uvall,
                clustercells_tsneAll,
                clustercells_umapAll,
            )

            # get images of cluster ids mapped to mask
            list_of_cluster_imgs = cell_map(mask, clusterids_df, seg_n, options)

            # plots the cluster imgs for the best z plane
            print("Saving pngs of cluster plots by best focal plane...")
            # plot_imgs(
            #     filename,
            #     output_dir,
            #     i,
            #     maskchs,
            #     options,
            #     cluster_cell_imgu[bestz],
            #     cluster_cell_imgcov[bestz],
            #     cluster_cell_imguall[bestz],
            #     cluster_cell_imgtotal[bestz],
            #     cluster_cell_texture[bestz],
            #     cluster_cell_imgtsneAll[bestz],
            #     cluster_cell_imgumapAll[bestz],
            # )
            plot_imgs(
                filename,
                output_dir,
                i,
                maskchs,
                options,
                clusterids_df,
                list_of_cluster_imgs,
            )

    if options.get("debug"):
        print("Elapsed time for cluster img saving: ", time.monotonic() - stime)

    # post process
    # find max len - for now only two cluster methods
    min1, max1 = options.get("num_cellclusters")[1:]
    min2, max2 = options.get("num_shapeclusters")[1:]
    a = max1 - min1
    b = max2 - min2
    d = max(a, b) + 1

    for i in all_clusters:
        if len(i) != d:
            i.extend([0] * (d - len(i)))

    all_clusters = np.array(all_clusters)
    options["cluster_types"] = types_list

    write_2_file(
        all_clusters,
        filename + "clustering" + options.get("num_cellclusters")[0] + "Scores",
        im,
        output_dir,
        cellidx,
        options,
    )


def make_DOT(mc, fc, coeffs, ll):
    pass


def powerset(iterable: List[int]):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def read_options_single(options_path: Path) -> Dict[str, Union[int, str]]:
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


def read_options(*options_paths: Path) -> Dict[str, Union[int, str]]:
    return ChainMap(*(read_options_single(path) for path in options_paths))


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
        if options.get("debug"):
            print("Output directory exists")
    else:
        path.mkdir()
        if options.get("debug"):
            print("Output directory created")


def showlegend(markernames: List[str], markertable: np.ndarray, outputfile: str, output_dir: Path):
    mmin = []
    mmax = []
    for k in range(0, len(markernames)):
        mmin.append(min(markertable[:, k]))
        mmax.append(max(markertable[:, k]))
    for i in range(0, len(markertable)):
        tplot = [
            (markertable[i, k] - mmin[k]) / (mmax[k] - mmin[k]) for k in range(0, len(markernames))
        ]
        plt.plot(tplot, label=["cluster " + str(i)])
    plt.xlabel("-".join(markernames))
    plt.ylabel("Relative value")
    plt.legend()
    plt.tick_params(axis="x", which="both", bottom=False, top=False)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_ticks([])
    # plt.show(block=False)
    plt.savefig(output_dir / outputfile, **figure_save_params)
    plt.clf()
    plt.close()


def find_locations(arr: np.ndarray) -> (np.ndarray, List[np.ndarray]):
    coords = np.indices(arr.shape).reshape(arr.ndim, arr.size)
    arr = arr.ravel()
    order = np.argsort(arr)
    arr = arr[order]
    coords = coords[:, order]
    locs = np.flatnonzero(np.diff(arr, prepend=arr[0] - 1))
    return arr[locs], np.split(coords, locs[1:], axis=1)


def get_last2d(data: np.ndarray, bestz: list, options: dict) -> np.ndarray:
    if options.get("image_dimension") == "3D" or data.ndim <= 2:
        return data

    bestz = bestz[0]
    slc = [0] * (data.ndim - 3)
    slc += [bestz, slice(None), slice(None)]
    return data[tuple(slc)]


def multidim_intersect(arr1, arr2):
    a = set((tuple(i) for i in arr1))
    b = set((tuple(i) for i in arr2))

    return np.array(list(a - b)).T


def find_cytoplasm(ROI_coords):
    cell_coords = ROI_coords[0]
    nucleus_coords = ROI_coords[1]
    cell_boundary_coords = ROI_coords[2]
    nuclei_boundary_coords = ROI_coords[3]

    cell_coords = [x.T for x in cell_coords]
    nucleus_coords = [x.T for x in nucleus_coords]
    cbc = [x.T for x in cell_boundary_coords]
    nbc = [x.T for x in nuclei_boundary_coords]
    # cell_coords = list(map(lambda x: x.T, cell_coords))
    # nucleus_coords = list(map(lambda x: x.T, nucleus_coords))

    cytoplasm_raw = [multidim_intersect(x, y) for x, y in zip(cell_coords, nucleus_coords)]
    cytoplasm_raw = [x.T for x in cytoplasm_raw]
    cytoplasm_cbc = [multidim_intersect(x, y) for x, y in zip(cytoplasm_raw, cbc)]
    cytoplasm_cbc = [x.T for x in cytoplasm_cbc]
    cytoplasm = [multidim_intersect(x, y) for x, y in zip(cytoplasm_cbc, nbc)]

    # cytoplasm = []
    # for i in range(1, len(cell_coords)):
    #     cytoplasm.append(self.multidim_intersect())
    # cytoplasm = list(map(lambda x, y: self.multidim_intersect(x, y), cell_coords, nucleus_coords))

    return cytoplasm


def quality_measures(
    im_list: List[IMGstruct],
    mask_list: List[MaskStruct],
    seg_metric_list: List[Dict[str, Any]],
    cell_total: List[int],
    img_files: List[Path],
    output_dir: Path,
    options: Dict[str, Any],
):
    """
    Quality Measurements for SPRM analysis
    """

    for i in range(len(img_files)):
        print("Writing out quality measures...")

        struct = dict()

        img_name = img_files[i].name
        im = im_list[i]
        mask = mask_list[i]

        bestz = mask.get_bestz()
        ROI_coords = mask.get_ROI()

        im_data = im.get_data()
        im_dims = im_data.shape

        im_channels = im_data[0, 0, :, bestz, :, :]
        pixels = im_dims[-2] * im_dims[-1]
        bgpixels = ROI_coords[0][0]
        cells = ROI_coords[0][1:]
        nuclei = ROI_coords[1][1:]

        # Image Quality Metrics that require cell segmentation
        struct["Image Quality Metrics that require cell segmentation"] = dict()
        if options.get("sprm_segeval_both") == 1:
            struct["Segmentation Evaluation Metrics"] = seg_metric_list[i][0]
            continue
        if options.get("sprm_segeval_both") == 2:
            struct["Segmentation Evaluation Metrics"] = seg_metric_list[i][0]

        channels = im.get_channel_labels()
        # get cytoplasm coords
        # cytoplasm = find_cytoplasm(ROI_coords)
        total_intensity_per_chan = im_channels[0, :, :, :]
        total_intensity_per_chan = np.reshape(
            total_intensity_per_chan,
            (
                total_intensity_per_chan.shape[0],
                total_intensity_per_chan.shape[1] * total_intensity_per_chan.shape[2],
            ),
        )
        total_intensity_per_chan = np.sum(total_intensity_per_chan, axis=1)

        # check / filter out 1-D coords - hot fix
        # cytoplasm_ndims = [x.ndim for x in cytoplasm]
        # cytoplasm_ndims = np.asarray(cytoplasm_ndims)
        # idx_ndims = np.where(cytoplasm_ndims == 1)[0]
        #
        # cytoplasm = np.delete(cytoplasm, idx_ndims).tolist()

        # cell total intensity per channel
        # total_intensity_path = output_dir / (img_name + '-cell_channel_total.csv')
        # total_intensity_file = get_paths(total_intensity_path)
        # total_intensity = pd.read_csv(total_intensity_file[0]).to_numpy()
        total_intensity_cell = np.concatenate(cells, axis=1)
        total_intensity_per_chancell = np.sum(
            im_channels[0, :, total_intensity_cell[0], total_intensity_cell[1]], axis=0
        )
        # total_intensity_per_chancell = np.sum(total_intensity[:, 1:], axis=0)

        total_intensity_per_chanbg = np.sum(im_channels[0, :, bgpixels[0], bgpixels[1]], axis=0)

        # total_intensity_nuclei_path = output_dir / (img_name + '-nuclei_channel_total.csv')
        # total_intensity_nuclei_file = get_paths(total_intensity_nuclei_path)
        # total_intensity_nuclei = pd.read_csv(total_intensity_nuclei_file[0]).to_numpy()
        # total_intensity_nuclei_per_chan = np.sum(total_intensity_nuclei[:, 1:], axis=0)

        # nuclei total intensity per channel
        total_intensity_nuclei = np.concatenate(nuclei, axis=1)
        total_intensity_nuclei_per_chan = np.sum(
            im_channels[0, :, total_intensity_nuclei[0], total_intensity_nuclei[1]], axis=0
        )

        # cytoplasm total intensity per channel
        # cytoplasm_all = np.concatenate(cytoplasm, axis=1)
        # total_cytoplasm = np.sum(im_channels[0, :, cytoplasm_all[0], cytoplasm_all[1]], axis=0)

        # nuc_cyto_avgR = total_intensity_nuclei_per_chan / total_cytoplasm
        nuc_cell_avgR = total_intensity_nuclei_per_chan / total_intensity_per_chancell
        cell_bg_avgR = (
            total_intensity_per_chancell
            / (total_intensity_per_chanbg / bgpixels.shape[1])
            / cell_total[i]
        )

        # read in silhouette scores
        sscore_path = output_dir / (img_name + "clusteringsilhouetteScores.csv")
        sscore_file = get_paths(sscore_path)
        sscore = pd.read_csv(sscore_file[0]).to_numpy()
        mean_all = sscore[0, 1:]
        maxscore = np.max(mean_all)
        maxidx = np.argmax(mean_all) + 2

        # read in signal csv
        signal_path = output_dir / (img_name + "-SNR.csv")
        signal_file = get_paths(signal_path)
        SNR = pd.read_csv(signal_file[0]).to_numpy()
        zscore = SNR[0, 1:]
        otsu = SNR[1, 1:]

        # Image Information
        struct["Image Information"] = dict()
        # Image Quality Metrics not requiring image segmentation
        struct["Image Quality Metrics not requiring image segmentation"] = dict()
        # Image Quality Metrics requiring background segmentation
        struct["Image Quality Metrics requiring background segmentation"] = dict()

        if not options.get("sprm_segeval_both") == 0:
            struct["Image Quality Metrics requiring background segmentation"][
                "Fraction of Pixels in Image Background"
            ] = seg_metric_list[i][1]
            struct["Image Quality Metrics requiring background segmentation"][
                "1/AvgCVBackground"
            ] = seg_metric_list[i][2]
            struct["Image Quality Metrics requiring background segmentation"][
                "FractionOfFirstPCBackground"
            ] = seg_metric_list[i][3]

        # Image Quality Metrics that require cell segmentation
        struct["Image Quality Metrics that require cell segmentation"][
            "Channel Statistics"
        ] = dict()
        struct["Image Information"]["Number of Channels"] = len(channels)
        struct["Image Quality Metrics that require cell segmentation"][
            "Number of Cells"
        ] = cell_total[i]
        # struct["Number of Background Pixels"] = bgpixels.shape[1]
        struct["Image Quality Metrics that require cell segmentation"][
            "Fraction of Image Occupied by Cells"
        ] = (pixels - bgpixels.shape[1]) / pixels

        struct["Image Quality Metrics not requiring image segmentation"][
            "Total Intensity"
        ] = dict()
        struct["Image Quality Metrics that require cell segmentation"]["Channel Statistics"][
            "Average per Cell Ratios"
        ] = dict()
        struct["Image Quality Metrics that require cell segmentation"][
            "Silhouette Scores From Clustering"
        ] = dict()
        struct["Image Quality Metrics that require cell segmentation"][
            "Silhouette Scores From Clustering"
        ]["Mean-All"] = dict()
        struct["Image Quality Metrics that require cell segmentation"][
            "Silhouette Scores From Clustering"
        ]["Max Silhouette Score"] = maxscore
        struct["Image Quality Metrics that require cell segmentation"][
            "Silhouette Scores From Clustering"
        ]["Cluster with Max Score"] = maxidx

        struct["Image Quality Metrics not requiring image segmentation"][
            "Signal To Noise Otsu"
        ] = dict()
        struct["Image Quality Metrics not requiring image segmentation"][
            "Signal To Noise Z-Score"
        ] = dict()

        for j in range(len(mean_all)):
            struct["Image Quality Metrics that require cell segmentation"][
                "Silhouette Scores From Clustering"
            ]["Mean-All"][j + 2] = mean_all[j]

        for j in range(len(channels)):
            struct["Image Quality Metrics not requiring image segmentation"]["Total Intensity"][
                channels[j]
            ] = total_intensity_per_chan[j]
            struct["Image Quality Metrics that require cell segmentation"]["Channel Statistics"][
                "Average per Cell Ratios"
            ][channels[j]] = dict()

            # struct["Image Quality Metrics not requiring image segmentation"]["Total Intensity"][
            #     channels[j]
            # ]["Cells"] = int(total_intensity_per_chancell[j])
            # struct["Image Quality Metrics not requiring image segmentation"]["Total Intensity"][
            #     channels[j]
            # ]["Background"] = total_intensity_per_chanbg[j]

            struct["Image Quality Metrics that require cell segmentation"]["Channel Statistics"][
                "Average per Cell Ratios"
            ][channels[j]]["Nuclear / Cell"] = nuc_cell_avgR[j]
            struct["Image Quality Metrics that require cell segmentation"]["Channel Statistics"][
                "Average per Cell Ratios"
            ][channels[j]]["Cell / Background"] = cell_bg_avgR[j]

            struct["Image Quality Metrics not requiring image segmentation"][
                "Signal To Noise Otsu"
            ][channels[j]] = otsu[j]
            struct["Image Quality Metrics not requiring image segmentation"][
                "Signal To Noise Z-Score"
            ][channels[j]] = zscore[j]

        with open(output_dir / (img_name + "-SPRM_Image_Quality_Measures.json"), "w") as json_file:
            json.dump(struct, json_file, indent=4, sort_keys=True, cls=NumpyEncoder)


def quality_measures_3D(
    im_list: List[IMGstruct],
    mask_list: List[MaskStruct],
    seg_metric_list: List[Dict[str, Any]],
    cell_total: List[int],
    img_files: List[Path],
    output_dir: Path,
    options: Dict[str, Any],
):
    """
    Quality Measurements for SPRM analysis
    """

    for i in range(len(img_files)):
        print("Writing out quality measures...")

        struct = dict()

        img_name = img_files[i].name
        im = im_list[i]
        mask = mask_list[i]

        bestz = mask.get_bestz()
        ROI_coords = mask.get_ROI()

        im_data = im.get_data()
        im_dims = im_data.shape

        im_channels = im_data[0, 0, :, bestz, :, :]
        pixels = im_dims[-2] * im_dims[-1]
        bgpixels = ROI_coords[0][0]
        cells = ROI_coords[0][1:]
        nuclei = ROI_coords[1][1:]

        # Image Quality Metrics that require cell segmentation
        struct["Image Quality Metrics that require cell segmentation"] = dict()
        if options.get("sprm_segeval_both") == 1:
            struct["Segmentation Evaluation Metrics"] = seg_metric_list[i][0]
            continue
        if options.get("sprm_segeval_both") == 2:
            struct["Segmentation Evaluation Metrics"] = seg_metric_list[i][0]

        channels = im.get_channel_labels()
        # get cytoplasm coords
        # cytoplasm = find_cytoplasm(ROI_coords)

        # dims are z c y x
        total_intensity_per_chan = im_channels
        total_intensity_per_chan = np.reshape(
            total_intensity_per_chan,
            (
                total_intensity_per_chan.shape[1],
                total_intensity_per_chan.shape[0]
                * total_intensity_per_chan.shape[2]
                * total_intensity_per_chan.shape[3],
            ),
        )
        total_intensity_per_chan = np.sum(total_intensity_per_chan, axis=1)

        # check / filter out 1-D coords - hot fix
        # cytoplasm_ndims = [x.ndim for x in cytoplasm]
        # cytoplasm_ndims = np.asarray(cytoplasm_ndims)
        # idx_ndims = np.where(cytoplasm_ndims == 1)[0]
        #
        # cytoplasm = np.delete(cytoplasm, idx_ndims).tolist()

        # cell total intensity per channel
        # total_intensity_path = output_dir / (img_name + '-cell_channel_total.csv')
        # total_intensity_file = get_paths(total_intensity_path)
        # total_intensity = pd.read_csv(total_intensity_file[0]).to_numpy()
        total_intensity_cell = np.concatenate(cells, axis=1)
        total_intensity_per_chancell = np.sum(
            im_channels[
                total_intensity_cell[0], :, total_intensity_cell[1], total_intensity_cell[2]
            ],
            axis=0,
        )
        # total_intensity_per_chancell = np.sum(total_intensity[:, 1:], axis=0)

        total_intensity_per_chanbg = np.sum(
            im_channels[bgpixels[0], :, bgpixels[1], bgpixels[2]], axis=0
        )

        # total_intensity_nuclei_path = output_dir / (img_name + '-nuclei_channel_total.csv')
        # total_intensity_nuclei_file = get_paths(total_intensity_nuclei_path)
        # total_intensity_nuclei = pd.read_csv(total_intensity_nuclei_file[0]).to_numpy()
        # total_intensity_nuclei_per_chan = np.sum(total_intensity_nuclei[:, 1:], axis=0)

        # nuclei total intensity per channel
        total_intensity_nuclei = np.concatenate(nuclei, axis=1)
        total_intensity_nuclei_per_chan = np.sum(
            im_channels[
                total_intensity_nuclei[0], :, total_intensity_nuclei[1], total_intensity_nuclei[2]
            ],
            axis=0,
        )

        # cytoplasm total intensity per channel
        # cytoplasm_all = np.concatenate(cytoplasm, axis=1)
        # total_cytoplasm = np.sum(im_channels[0, :, cytoplasm_all[0], cytoplasm_all[1]], axis=0)

        # nuc_cyto_avgR = total_intensity_nuclei_per_chan / total_cytoplasm
        nuc_cell_avgR = total_intensity_nuclei_per_chan / total_intensity_per_chancell
        cell_bg_avgR = (
            total_intensity_per_chancell
            / (total_intensity_per_chanbg / bgpixels.shape[1])
            / cell_total[i]
        )

        # read in silhouette scores
        sscore_path = output_dir / (img_name + "clusteringsilhouetteScores.csv")
        sscore_file = get_paths(sscore_path)
        sscore = pd.read_csv(sscore_file[0]).to_numpy()
        mean_all = sscore[0, 1:]
        maxscore = np.max(mean_all)
        maxidx = np.argmax(mean_all) + 2

        # read in signal csv
        signal_path = output_dir / (img_name + "-SNR.csv")
        signal_file = get_paths(signal_path)
        SNR = pd.read_csv(signal_file[0]).to_numpy()
        zscore = SNR[0, 1:]
        otsu = SNR[1, 1:]

        # Image Information
        struct["Image Information"] = dict()
        # Image Quality Metrics not requiring image segmentation
        struct["Image Quality Metrics not requiring image segmentation"] = dict()
        # Image Quality Metrics requiring background segmentation
        struct["Image Quality Metrics requiring background segmentation"] = dict()

        if not options.get("sprm_segeval_both") == 0:
            struct["Image Quality Metrics requiring background segmentation"][
                "Fraction of Pixels in Image Background"
            ] = seg_metric_list[i][1]
            struct["Image Quality Metrics requiring background segmentation"][
                "1/AvgCVBackground"
            ] = seg_metric_list[i][2]
            struct["Image Quality Metrics requiring background segmentation"][
                "FractionOfFirstPCBackground"
            ] = seg_metric_list[i][3]

        # Image Quality Metrics that require cell segmentation
        struct["Image Quality Metrics that require cell segmentation"][
            "Channel Statistics"
        ] = dict()
        struct["Image Information"]["Number of Channels"] = len(channels)
        struct["Image Quality Metrics that require cell segmentation"][
            "Number of Cells"
        ] = cell_total[i]
        # struct["Number of Background Pixels"] = bgpixels.shape[1]
        struct["Image Quality Metrics that require cell segmentation"][
            "Fraction of Image Occupied by Cells"
        ] = (pixels - bgpixels.shape[1]) / pixels

        struct["Image Quality Metrics not requiring image segmentation"][
            "Total Intensity"
        ] = dict()
        struct["Image Quality Metrics that require cell segmentation"]["Channel Statistics"][
            "Average per Cell Ratios"
        ] = dict()
        struct["Image Quality Metrics that require cell segmentation"][
            "Silhouette Scores From Clustering"
        ] = dict()
        struct["Image Quality Metrics that require cell segmentation"][
            "Silhouette Scores From Clustering"
        ]["Mean-All"] = dict()
        struct["Image Quality Metrics that require cell segmentation"][
            "Silhouette Scores From Clustering"
        ]["Max Silhouette Score"] = maxscore
        struct["Image Quality Metrics that require cell segmentation"][
            "Silhouette Scores From Clustering"
        ]["Cluster with Max Score"] = maxidx

        struct["Image Quality Metrics not requiring image segmentation"][
            "Signal To Noise Otsu"
        ] = dict()
        struct["Image Quality Metrics not requiring image segmentation"][
            "Signal To Noise Z-Score"
        ] = dict()

        for j in range(len(mean_all)):
            struct["Image Quality Metrics that require cell segmentation"][
                "Silhouette Scores From Clustering"
            ]["Mean-All"][j + 2] = mean_all[j]

        for j in range(len(channels)):
            struct["Image Quality Metrics not requiring image segmentation"]["Total Intensity"][
                channels[j]
            ] = total_intensity_per_chan[j]
            struct["Image Quality Metrics that require cell segmentation"]["Channel Statistics"][
                "Average per Cell Ratios"
            ][channels[j]] = dict()

            # struct["Image Quality Metrics not requiring image segmentation"]["Total Intensity"][
            #     channels[j]
            # ]["Cells"] = int(total_intensity_per_chancell[j])
            # struct["Image Quality Metrics not requiring image segmentation"]["Total Intensity"][
            #     channels[j]
            # ]["Background"] = total_intensity_per_chanbg[j]

            struct["Image Quality Metrics that require cell segmentation"]["Channel Statistics"][
                "Average per Cell Ratios"
            ][channels[j]]["Nuclear / Cell"] = nuc_cell_avgR[j]
            struct["Image Quality Metrics that require cell segmentation"]["Channel Statistics"][
                "Average per Cell Ratios"
            ][channels[j]]["Cell / Background"] = cell_bg_avgR[j]

            struct["Image Quality Metrics not requiring image segmentation"][
                "Signal To Noise Otsu"
            ][channels[j]] = otsu[j]
            struct["Image Quality Metrics not requiring image segmentation"][
                "Signal To Noise Z-Score"
            ][channels[j]] = zscore[j]

        with open(output_dir / (img_name + "-SPRM_Image_Quality_Measures.json"), "w") as json_file:
            json.dump(struct, json_file, indent=4, sort_keys=True, cls=NumpyEncoder)


def check_shape(im, mask):
    # put in check here for matching dims

    return (
        im.get_data().shape[4] != mask.get_data().shape[4]
        or im.get_data().shape[5] != mask.get_data().shape[5]
    )


def reallocate_parallel(im, mask, ichan, options):
    pass


def reallocate_and_merge_intensities(
    im: IMGstruct,
    mask: MaskStruct,
    optional_img_file: Optional[Path],
    options: Dict[str, Any],
):
    # check if image needs to be resized to match mask
    if check_shape(im, mask):
        s, t, c, z, y, x = im.get_data().shape

        # allocate memory for reallocated im
        newim = np.zeros((s, t, c, z, mask.get_data().shape[4], mask.get_data().shape[5]))

        ROI = mask.get_data()[0, 0, 0, 0, :, :]  # assume chan 0 is the cell mask
        IMSimg = im.get_data()[0, 0, 0, 0, :, :]

        # do findpixelpixelfraction and getrelevantpixel index once
        print("START findpixelfraction...")
        X, A, cellArea, reducedsize = findpixelfractions(
            ROI.reshape(-1), ROI.shape, IMSimg.shape, c
        )
        # print(cellArea)
        # exit()

        print("END findpixelfraction...")
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

# def recluster(output_dir: Path, im: IMGstruct, options: Dict):
#     filename = 'Recluster'
#
#     # features
#     meanV = '*-cell_channel_mean.csv'
#     covar = '*-cell_channel_covar.csv'
#     totalV = '*-cell_channel_total.csv'
#     # meanC = '*-cell_channel_meanAllChannels.csv'
#     shapeV = '*-cell_shape.csv'
#
#     # read in and concatenate all feature csvs
#     meanAll = get_paths(output_dir / meanV)
#     covarAll = get_paths(output_dir / covar)
#     totalAll = get_paths(output_dir / totalV)
#     # meanCAll = get_paths(output_dir / meanC)
#     shapeAll = get_paths(output_dir / shapeV)
#
#     for i in range(0, len(meanAll)):
#         meanAll_read = pd.read_csv(meanAll[i])
#         covarAll_read = pd.read_csv(covarAll[i])
#         totalAll_read = pd.read_csv(totalAll[i])
#         # meanCAll_read = pd.read_csv(meanCAll[i])
#         shapeAll_read = pd.read_csv(shapeAll[i])
#
#         if i == 0:
#             meanAll_pd = meanAll_read
#             covarAll_pd = covarAll_read
#             totalAll_pd = totalAll_read
#             # meanCALL_pd = meanCAll_read
#             shapeAll_pd = shapeAll_read
#         else:
#             meanAll_pd = pd.concat([meanAll_pd, meanAll_read], axis=1, sort=False)
#             covarAll_pd = pd.concat([covarAll_pd, covarAll_read], axis=1, sort=False)
#             totalAll_pd = pd.concat([totalAll_pd, totalAll_read], axis=1, sort=False)
#             # meanCALL_pd = pd.concat([meanCALL_pd, meanCAll_read], axis=1, sort=False)
#             shapeAll_pd = pd.concat([shapeAll_pd, shapeAll_read], axis=1, sort=False)
#
#     meanAll_np = meanAll_pd.to_numpy()
#     covarAll_np = covarAll_pd.to_numpy()
#     totalAll_np = totalAll_pd.to_numpy()
#     # meanCALL_np = meanCALL_pd.to_numpy()
#     shapeAll_np = shapeAll_pd.to_numpy()
#
#     print('Reclustering cells and getting back the labels and centers...')
#     clustercells_uv, clustercells_uvcenters = cell_cluster(meanAll_np, options)
#     clustercells_cov, clustercells_covcenters = cell_cluster(covarAll_np, options)
#     clustercells_total, clustercells_totalcenters = cell_cluster(totalAll_np, options)
#     clustercells_uvall, clustercells_uvallcenters = cell_cluster(meanAll_np, options)
#     clustercells_shapevectors, shapeclcenters = shape_cluster(shapeAll_np, options)
#
#     print('Making legend for the recluster...')
#     make_legends(im, filename, output_dir, options, clustercells_uvcenters, clustercells_covcenters,
#                  clustercells_totalcenters, clustercells_uvallcenters,
#                  shapeclcenters)
#
#     print('Writing out all cell cluster IDs for recluster cells...')
#     cell_cluster_IDs(filename, output_dir, options, clustercells_uv, clustercells_cov, clustercells_total,
#                      clustercells_uvall,
#                      clustercells_shapevectors)


def quality_control(mask: MaskStruct, img: IMGstruct, ROI_coords: List, options: Dict):
    # best z +- from options
    set_zdims(mask, img, options)

    # find cells on edge
    find_edge_cells(mask)

    # check to see if image is normalized aka no negative values
    normalize_image(img)

    # normalize bg intensities
    if options.get("normalize_bg"):
        normalize_background(img, mask, ROI_coords)


def normalize_image(img):
    """
    basic normalization of averaging negative values and setting that as background
    """
    if (img.data < 0).any():
        # find a boolean matrix of negative values
        nbool = img.data < 0

        # get average of negative and set that as the background or 0
        n_avg = -np.average(img.data[nbool])
        img.data = img.data + n_avg

        # everything that is < 0 is background
        img.data[img.data < 0] = 0
        return
    else:
        return


def normalize_background(im, mask, ROI_coords):
    # pass
    img = im.get_data().copy()
    # img = img.astype("int64")
    dims = mask.get_data().shape

    if dims[3] > 1:  # 3D
        for i in range(img.shape[2]):
            img_ch = img[0, 0, i, :, :, :]
            bg_img_ch = img_ch[
                ROI_coords[0][0][0, :], ROI_coords[0][0][1, :], ROI_coords[0][0][2, :]
            ]

            avg = np.sum(bg_img_ch) / ROI_coords[0][0].shape[1]

            if avg == 0:
                continue
            elif avg < 1:
                avg = 1

            img[0, 0, i, :, :, :] = img_ch / avg
    else:
        for i in range(img.shape[2]):
            img_ch = img[0, 0, i, 0, :, :]
            bg_img_ch = img_ch[ROI_coords[0][0][0, :], ROI_coords[0][0][1, :]]

            avg = np.sum(bg_img_ch) / ROI_coords[0][0].shape[1]

            if avg == 0:
                continue
            elif avg < 1:
                avg = 1

            img[0, 0, i, 0, :, :] = img_ch / avg

    im.set_data(img)


def set_zdims(mask: MaskStruct, img: IMGstruct, options: Dict):
    bound = options.get("zslices")
    bestz = mask.get_bestz()
    z = mask.get_data().shape[3]

    lower_bound = bestz[0] - bound
    upper_bound = bestz[0] + bound + 1
    if lower_bound < 0 or upper_bound > z:
        raise ValueError("zslice bound is invalid. Please reset and run SPRM")
    elif z > 1:
        idx = np.arange(0, z).tolist()
        mask.set_bestz(idx)
        return
    else:
        new_mask = mask.get_data()[:, :, :, lower_bound:upper_bound, :, :]
        new_img = img.get_data()[:, :, :, lower_bound:upper_bound, :, :]

        mask.set_data(new_mask)
        mask.set_bestz([0])  # set best z back to 0 since we deleted some zstacks
        img.set_data(new_img)


def find_edge_cells(mask):
    z = mask.get_data().shape[3]
    channels = mask.get_data().shape[2]
    data = mask.get_data()[0, 0, 0, :, :, :]
    border = []

    if z > 1:  # 3D case
        for i in range(0, z):
            border += list(data[i, 0, :-1])  # Top row (left to right), not the last element.
            border += list(data[i, :-1, -1])  # Right column (top to bottom), not the last element.
            border += list(data[i, -1, :0:-1])  # Bottom row (right to left), not the last element.
            border += list(data[i, ::-1, 0])  # Left column (bottom to top), all elements element.

    else:  # 2D case
        bestz = mask.get_bestz()[0]
        data = mask.get_data()[0, 0, :, bestz, :, :]

        for i in range(0, channels):
            border += list(data[i, 0, :-1])  # Top row (left to right), not the last element.
            border += list(data[i, :-1, -1])  # Right column (top to bottom), not the last element.
            border += list(data[i, -1, :0:-1])  # Bottom row (right to left), not the last element.
            border += list(data[i, ::-1, 0])  # Left column (bottom to top), all elements element.

    border = np.unique(border).tolist()
    mask.set_edge_cells(border)

    sMask = mask.get_data()[0, 0, 0, :, :, :]
    unique = np.unique(sMask)[1:]
    og_cell_idx = mask.get_cell_index()

    # 3D case
    # if z > 1:
    #     pass
    # else:  # 2D case

    if (og_cell_idx == unique).all():
        interiorCells = [i for i in unique if i not in border and i not in mask.get_bad_cells()]
        mask.set_interior_cells(interiorCells)
        mask.set_cell_index(interiorCells)
    else:
        og_in_cell_idx = [
            og
            for og, new in zip(og_cell_idx, unique)
            if new not in border and new not in mask.get_bad_cells()
        ]
        interiorCells = [i for i in unique if i not in border and i not in mask.get_bad_cells()]
        mask.set_interior_cells(interiorCells)
        mask.set_cell_index(og_in_cell_idx)


# @nb.njit()
# def nb_ROI_crop(t: (np.ndarray, np.ndarray), imgog: np.ndarray) -> np.ndarray:
#
#     # a = np.empty((2, imgog.shape[2]), dtype=np.ndarray)
#     for i in range(2):
#         xmax, xmin, ymax, ymin = np.max(t[i][0]), np.min(t[i][0]), np.max(t[i][1]), np.min(t[i][1])
#
#         for k in range(0, imgog.shape[2]):
#
#             imgroi = imgog[0, 0, k, 0, xmin:xmax+1, ymin:ymax+1]
#             imgroi = (imgroi / imgroi.max()) * 255
#             imgroi = imgroi.astype(np.uint8)
#
#             # a[i, k] = imgroi
#
#     return a

# @nb.njit(parallel=True)
# def abc(imga, cl, curROI, xmax, xmin, ymax, ymin):
#
#     channellist = nbList()
#
#     for j in nb.prange(cl):  # For each channel
#         # img = im.get_data()[0, 0, j, bestz[0], :, :]
#         # img = img[xmin:xmax+1, ymin:ymax+1]
#         # img = np.multiply(interiormask, img)
#         # convert to uint
#         imgroi = imga[0, 0, j, 0, curROI[0], curROI[1]]
#         index = np.arange(imgroi.shape[0])
#
#         # make cropped 2D image
#         img = np.zeros((xmax - xmin + 1, ymax - ymin + 1))
#         xn = curROI[1] - xmin
#         yn = curROI[0] - ymin
#         img[xn, yn] = imgroi[index]
#         img = (img / img.max()) * 255
#         img = img.astype(np.uint8)
#
#         channellist.append(img)
#
#     return channellist


def glcm(
    im,
    mask,
    output_dir,
    filename,
    options,
    angle,
    distances,
    ROI_coords,
):
    """
    By: Young Je Lee and Ted Zhang
    """

    # texture_all=[]
    # header = []
    cellidx = mask.get_cell_index()

    colIndex = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    inCells = mask.get_interior_cells().copy()
    texture_all = np.zeros(
        (2, len(inCells), im.get_data().shape[2], len(colIndex) * len(distances))
    )

    # get headers
    channel_labels = im.get_channel_labels().copy() * len(distances) * len(colIndex) * 2
    maskn = (
        mask.get_channel_labels()[0:2] * im.get_data().shape[2] * len(distances) * len(colIndex)
    )
    maskn.sort()
    cols = colIndex * im.get_data().shape[2] * len(distances) * 2
    dlist = distances * im.get_data().shape[2] * len(colIndex) * 2
    column_format = ["{channeln}_{mask}: {col}-{d}" for i in range(len(channel_labels))]
    header = list(
        map(
            lambda x, y, z, c, t: x.format(channeln=y, mask=z, col=c, d=t),
            column_format,
            channel_labels,
            maskn,
            cols,
            dlist,
        )
    )

    # ogimg = im.get_data()
    # cellsbychannelROI = np.empty((2, cell_total[0], im.get_data().shape[2]), dtype=np.ndarray)
    # inCells = np.asarray(inCells)

    # masklistx = nbList()
    # masklisty = nbList()
    # for i in range(0, 2):
    #     celllisty = nbList()
    #     celllistx = nbList()
    #     for j in range(0, len(ROI_coords[0])):
    #         celllistx.append(nbList(ROI_coords[i][j][0].tolist()))
    #         celllisty.append(nbList(ROI_coords[i][j][1].tolist()))
    # masklistx.append(celllistx)
    # masklisty.append(celllisty)

    # l = []
    # for j in range(len(inCells)):
    #     # a = np.empty((2, imgog.shape[2]), dtype=np.ndarray)
    #     t = ROI_coords[0][inCells[j]], ROI_coords[1][inCells[j]]
    #     a = nb_ROI_crop(t, ogimg)
    # l.append(a)

    # nblist = nb_ROI_crop(masklistx, masklisty, inCells, ogimg)

    for i in range(2):
        for idx in range(len(inCells)):  # For each cell
            curROI = ROI_coords[i][inCells[idx]]

            if curROI.size == 0:
                continue

            xmax, xmin, ymax, ymin = (
                np.max(curROI[1]),
                np.min(curROI[1]),
                np.max(curROI[0]),
                np.min(curROI[0]),
            )

            imga = im.get_data()
            # cl = len(im.channel_labels)

            # l = abc(imga, cl, curROI, xmax, xmin, ymax, ymin)

            for j in range(len(im.channel_labels)):  # For each channel
                # filter by SNR: Z-Score < 1: texture_all[:, :, j, :] = 0
                # continue

                # convert to uint
                imgroi = imga[0, 0, j, 0, curROI[0], curROI[1]]
                index = np.arange(imgroi.shape[0])

                # make cropped 2D image
                img = np.zeros((xmax - xmin + 1, ymax - ymin + 1))
                xn = curROI[1] - xmin
                yn = curROI[0] - ymin
                img[xn, yn] = imgroi[index]
                img = (img / img.max()) * 255
                img = img.astype(np.uint8)

                for d in range(len(distances)):
                    result = greycomatrix(
                        img, [distances[d]], [angle], levels=256
                    )  # Calculate GLCM
                    result = result[
                        1:, 1:
                    ]  # Remove background influence by delete first row & column

                    for ls in range(len(colIndex)):  # Get properties
                        texture_all[i, idx, j, d + ls] = greycoprops(
                            result, colIndex[ls]
                        ).flatten()[0]

    ctexture = np.concatenate(texture_all, axis=1)
    ctexture = ctexture.reshape(len(inCells), -1)

    # For csv writing
    write_2_csv(header, ctexture, filename + "_" + "texture", output_dir, cellidx, options)

    # add timepoint dim so that feature is in sync
    texture_all = texture_all[np.newaxis, :, :, :, :]

    return texture_all, header


def glcmProcedure(im, mask, output_dir, filename, ROI_coords, options):
    """
    Wrapper for GLCM
    """

    print("GLCM calculation initiated")

    angle = options.get("glcm_angles")
    distances = options.get("glcm_distances")
    angle = "".join(angle)[1:-1].split(",")
    distances = "".join(distances)[1:-1].split(",")
    angle = [int(i) for i in angle][0]  # Only supports 0 for now
    distances = [int(i) for i in distances]
    stime = time.monotonic()
    texture, texture_featureNames = glcm(
        im, mask, output_dir, filename, options, angle, distances, ROI_coords
    )
    print("GLCM calculations completed: " + str(time.monotonic() - stime))

    return [texture, texture_featureNames]


def DR_AllFeatures(all_clusters, types_list, filename, cellidx, output_dir, options, *argv):
    """

    By: Young Je Lee and Ted Zhang

    """

    # matrix_mean = argv[0]
    matrix_cov = argv[0]
    matrix_total = argv[1]
    matrix_meanAll = argv[2]

    # get features into correct shapes - cov, total
    # matrix_mean = matrix_mean[0]
    matrix_cov = matrix_cov[0]
    matrix_total = matrix_total[0]

    matrix_cov = matrix_cov.reshape(
        (matrix_cov.shape[0], matrix_cov.shape[1], matrix_cov.shape[2] * matrix_cov.shape[3])
    )
    matrix_cov = np.concatenate(matrix_cov, axis=1)

    matrix_total = matrix_total.reshape(
        (
            matrix_total.shape[0],
            matrix_total.shape[1],
            matrix_total.shape[2] * matrix_total.shape[3],
        )
    )
    matrix_total = np.concatenate(matrix_total, axis=1)

    if options.get("skip_outlinePCA"):
        matrix_texture = argv[3]
        matrix_shape = np.zeros((matrix_cov.shape[0], 100))
    else:
        matrix_shape = argv[3]
        matrix_texture = argv[4]

    tSNE_allfeatures_headers = []
    cmd = options.get("tSNE_all_preprocess")[1]
    perplexity = options.get("tSNE_all_perplexity")
    pcaMethod = options.get("tsne_all_svdsolver4pca")
    tSNEInitialization = options.get("tSNE_all_tSNEInitialization")
    numComp = options.get("tSNE_num_components")
    for i in range(numComp):
        tSNE_allfeatures_headers.append(str(i) + "th PC")
    n_iter = 1000
    learning_rate = 1
    matrix_texture = matrix_texture[:, : int(len(matrix_texture[0]) / 2)]

    if options.get("tSNE_texture_calculation_skip"):
        matrix_all_OnlyCell_original = np.concatenate(
            (matrix_cov, matrix_total, matrix_meanAll, matrix_shape),
            axis=1,
        )
    else:
        matrix_all_OnlyCell_original = np.concatenate(
            (matrix_cov, matrix_total, matrix_meanAll, matrix_shape, matrix_texture),
            axis=1,
        )

    tsne_ee = options.get("tSNE_all_ee")
    try:
        early_exaggeration = float(tsne_ee)
    except ValueError:
        if tsne_ee == "default":
            early_exaggeration = len(matrix_all_OnlyCell_original) / 10
        else:
            raise ValueError(f"Invalid tSNE_all_ee value: {tsne_ee}")

    matrix_all_OnlyCell = matrix_all_OnlyCell_original.copy()
    if cmd == "zscore":
        matrix_all_OnlyCell = np.asarray(matrix_all_OnlyCell, dtype=float)
        matrix_all_OnlyCell = stats.zscore(matrix_all_OnlyCell, axis=0)
        matrix_all_OnlyCell = np.array(matrix_all_OnlyCell)
        matrix_all_OnlyCell = matrix_all_OnlyCell[
            :, ~np.isnan(matrix_all_OnlyCell).any(axis=0)
        ]  # Remove NAN

    elif cmd == "blockwise_zscore":
        matrix_mean = BlockwiseZscore(matrix_mean)
        matrix_cov = BlockwiseZscore(matrix_cov)
        matrix_total = BlockwiseZscore(matrix_total)
        matrix_meanAll = BlockwiseZscore(matrix_meanAll)
        matrix_shape = BlockwiseZscore(matrix_shape)
        matrix_texture = BlockwiseZscore(matrix_texture)
        matrix_all_OnlyCell = np.concatenate(
            (matrix_mean, matrix_cov, matrix_total, matrix_meanAll, matrix_shape, matrix_texture),
            axis=1,
        )

    matrix_all_OnlyCell_full = matrix_all_OnlyCell.copy()
    # tol
    if tSNEInitialization == "pca":
        # try to solve full svd
        tries = 0
        while True:
            try:
                m = PCA(n_components=numComp, svd_solver=pcaMethod).fit(matrix_all_OnlyCell)
                break
            except Exception as e:
                print(e)
                print("Exceptions caught: ", tries)
                if tries == 0:
                    m = PCA(n_components=numComp, svd_solver="randomized").fit(matrix_all_OnlyCell)
                    tries += 1
                else:
                    print("halving the features in tSNE for PCA fit...")
                    n_samples = int(matrix_all_OnlyCell.shape[0] / 2)
                    idx = np.random.choice(
                        matrix_all_OnlyCell_full.shape[0], n_samples, replace=False
                    )
                    matrix_all_OnlyCell = matrix_all_OnlyCell_full[idx, :]

        matrix_all_OnlyCell = m.transform(matrix_all_OnlyCell_full)

    t_matrix_all_OnlyCell_full = matrix_all_OnlyCell.copy()
    # init tSNE
    tsne = TSNE(
        n_components=numComp,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_iter=n_iter,
        init="random",
        random_state=0,
    )

    while True:
        try:
            tsne_all_OnlyCell = tsne.fit_transform(matrix_all_OnlyCell)
            break
        except Exception as e:
            print(e)
            print("halving dataset in tSNE for tSNE fit...")
            n_samples = int(matrix_all_OnlyCell.shape[0] / 2)
            idx = np.random.choice(t_matrix_all_OnlyCell_full.shape[0], n_samples, replace=False)
            matrix_all_OnlyCell = t_matrix_all_OnlyCell_full[idx, :]

    # 2D - Scatterplot
    plt.scatter(tsne_all_OnlyCell[:, 0], tsne_all_OnlyCell[:, 1], marker=".")
    plt.savefig(output_dir / (filename + "-tSNE_allfeatures.png"), bbox_inches="tight")
    plt.clf()
    plt.close()

    header = [x for x in range(1, tsne_all_OnlyCell.shape[1] + 1)]
    write_2_csv(
        header, tsne_all_OnlyCell, filename + "-tSNE_allfeatures", output_dir, cellidx, options
    )

    clustercells_all, clustercells_allcenters = cell_cluster(
        tsne_all_OnlyCell,
        types_list,
        all_clusters,
        "tSNE_allfeatures",
        options,
        output_dir,
        cellidx,
    )
    # clusterMembership_descending=np.argsort(np.bincount(clustercells_all))
    # for i in range(len(clustercells_all)):
    #    clustercells_all[i]=len(clustercells_allcenters)-1-np.where(clusterMembership_descending==clustercells_all[i])[0]

    # umap
    reducer = umap.UMAP()
    umap_features = matrix_all_OnlyCell_full.copy()
    umap_embed = reducer.fit_transform(umap_features)

    # 2D umap
    plt.scatter(umap_embed[:, 0], umap_embed[:, 1], marker=".")
    plt.savefig(output_dir / (filename + "-UMAP_allfeatures.png"), bbox_inches="tight")
    plt.clf()
    plt.close()

    umap_header = [x for x in range(1, umap_embed.shape[1] + 1)]
    write_2_csv(
        umap_header, umap_embed, filename + "-UMAP_allfeatures", output_dir, cellidx, options
    )

    clustercells_all_umap, clustercells_allcenters_umap = cell_cluster(
        umap_embed, types_list, all_clusters, "UMAP_allfeatures", options, output_dir, cellidx
    )

    return (
        clustercells_all,
        clustercells_allcenters,
        tSNE_allfeatures_headers,
        clustercells_all_umap,
        clustercells_allcenters_umap,
    )


def BlockwiseZscore(data):
    data_zscored = stats.zscore(data, axis=0)
    data_zscored_nanRemoved = data_zscored[:, ~np.isnan(data_zscored).any(axis=0)]
    data_zscored = data_zscored_nanRemoved * (1 / len(data_zscored_nanRemoved[0]))
    return data_zscored
