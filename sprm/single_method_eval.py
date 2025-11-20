import importlib.resources
import logging
import pickle
import re
import xml.etree.ElementTree as ET
from math import prod
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

import aicsimageio
import numpy as np
import xmltodict
from PIL import Image
from pint import Quantity, UnitRegistry
from scipy.sparse import csr_matrix
from scipy.stats import variation
from skimage.filters import threshold_mean
from skimage.morphology import area_closing, closing, disk
from skimage.segmentation import morphological_geodesic_active_contour as MorphGAC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from threadpoolctl import threadpool_info
from pprint import pprint

"""
Companion to SPRM.py
Package functions that evaluate a single segmentation method
Author: Haoran Chen and Ted Zhang
Version: 1.5
09/08/2021
"""

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

schema_url_pattern = re.compile(r"\{(.+)\}OME")


def thresholding(img):
    threshold = threshold_mean(img.astype(np.int64))
    img_thre = img > threshold
    img_thre = img_thre * 1
    return img_thre


def fraction(img_bi, mask_bi):
    foreground_all = np.sum(img_bi)
    background_all = img_bi.shape[0] * img_bi.shape[1] - foreground_all
    mask_all = np.sum(mask_bi)
    background = len(np.where(mask_bi - img_bi == 1)[0])
    foreground = np.sum(mask_bi * img_bi)
    return foreground / foreground_all, background / background_all, foreground / mask_all


def foreground_separation(img_thre):
    img_thre = img_thre.astype(np.int16)
    print(f"foreground_sep point 1 {img_thre.shape} {img_thre.dtype}")
    contour_ref = img_thre.copy()
    print(f"foreground_sep point 2 {img_thre.shape} {img_thre.dtype}")
    img_thre = closing(img_thre, disk(1))

    img_thre = -img_thre + 1
    print(f"foreground_sep point 3 {img_thre.shape} {img_thre.dtype}")
    img_thre = closing(img_thre, disk(2))
    img_thre = -img_thre + 1

    print(f"foreground_sep point 4 {img_thre.shape} {img_thre.dtype}")
    img_thre = closing(img_thre, disk(20))

    img_thre = -img_thre + 1
    print(f"foreground_sep point 5 {img_thre.shape} {img_thre.dtype}")
    img_thre = closing(img_thre, disk(10))
    img_thre = -img_thre + 1

    print(f"foreground_sep point 6 {img_thre.shape} {img_thre.dtype}")
    img_thre = area_closing(img_thre, 20000, connectivity=2)
    contour_ref = contour_ref.astype(float)
    img_thre = img_thre.astype(float)
    print(f"foreground_sep point 7 {img_thre.shape} {img_thre.dtype}")
    img_binary = MorphGAC(
        -contour_ref + 1, 5, -img_thre + 1, smoothing=1, balloon=0.8, threshold=0.5
    )
    img_binary = img_binary.astype(np.int16)
    print(f"foreground_sep point 8 {img_binary.shape} {img_binary.dtype}")
    img_binary = area_closing(img_binary, 1000, connectivity=2)
    print(f"foreground_sep point 9 {img_binary.shape} {img_binary.dtype}")

    return -img_binary + 1


def uniformity_CV(is_foreground, img, bestz):
    CV = []
    for ch_idx, z_idx, channel in img.get_img_channel_generator(z=bestz[0]):
        channel = channel / np.mean(channel)
        intensity = channel[0,0,is_foreground]
        CV.append(np.std(intensity))
    return np.average(CV)


def uniformity_fraction(is_foreground, img, bestz) -> float:
    feature_matrix_pieces = []
    for ch_idx, z_idx, channel in img.get_img_channel_generator(z=bestz[0]):
        ss = StandardScaler()
        channel_z = ss.fit_transform(channel[0,0].copy())
        intensity = channel_z[is_foreground]
        feature_matrix_pieces.append(intensity)
    feature_matrix = np.vstack(feature_matrix_pieces)
    print(f"feature matrix: {feature_matrix.shape} {feature_matrix.dtype}")
    pca = PCA(n_components=1)
    model = pca.fit(feature_matrix.T)
    fraction = model.explained_variance_ratio_[0]
    print(f"uniformity_fraction returning {fraction}")
    return fraction


def foreground_uniformity(img_bi, mask, img, bestz):
    # foreground_loc = np.argwhere((img_bi - mask) == 1)
    is_foreground = (img_bi - mask) == 1
    CV = uniformity_CV(is_foreground, img, bestz)
    print(f"CV is {type(CV)} {CV}")
    foreground_pixel_num = is_foreground.sum()
    print(f"foreground_pixel_num = {foreground_pixel_num}")
    foreground_loc_fraction = 1
    fraction = None

    if foreground_pixel_num < 1:
        print("no pixels in the foreground")
        return CV, None

    print(f"img_bi is {img_bi.shape}, mask is {mask.shape}, bestz is {bestz}")
    while foreground_loc_fraction > 0:
        try:
            is_foreground_sampled = np.logical_and(
                is_foreground,
                np.random.choice([True, False],
                                 size=is_foreground.shape,
                                 p=[foreground_loc_fraction,
                                    1.0 - foreground_loc_fraction])
            )
            # foreground_loc_sampled = foreground_loc[
            #     np.random.randint(
            #         foreground_pixel_num,
            #         size=round(foreground_pixel_num * foreground_loc_fraction),
            #     ),
            #     :,
            # ]
            print(f"randomization worked with {foreground_loc_fraction}")
            fraction = uniformity_fraction(is_foreground_sampled, img, bestz)
            print(f"uniformity_fraction worked with {foreground_loc_fraction}")
            break
        except Exception as excp:
            print(f"got expected exception {excp}")
            foreground_loc_fraction = foreground_loc_fraction / 2
    print(f"CV: {CV}, fraction: {fraction}")
    return CV, fraction


def background_uniformity(img_bi, img, bestz):
    # background_loc = np.argwhere(img_bi == 0)
    is_background = (img_bi == 0)
    CV = uniformity_CV(is_background, img, bestz)
    print(f"CV for background is {CV}")

    background_pixel_num = is_background.sum()
    background_loc_fraction = 1
    fraction = None

    if background_pixel_num < 1:
        print("no pixels in the background")
        return CV, None

    while background_loc_fraction > 0:
        try:
            # background_loc_sampled = background_loc[
            #     np.random.randint(
            #         background_pixel_num,
            #         size=round(background_pixel_num * background_loc_fraction),
            #     ),
            #     :,
            # ]
            is_background_sampled = np.logical_and(
                is_background,
                np.random.choice([True, False],
                                 size=is_background.shape,
                                 p=[background_loc_fraction,
                                    1.0 - background_loc_fraction])
            )
            print(f"background randomization worked with {background_loc_fraction}")
            fraction = uniformity_fraction(is_background_sampled, img, bestz)
            print(f"background uniformity_fraction worked with {background_loc_fraction}")
            break
        except Exception as excp:
            print(f"bacground got expected exception {excp}")
            background_loc_fraction = background_loc_fraction / 2
    return CV, fraction


def cell_uniformity_CV(feature_matrix):
    CV = []
    for i in range(feature_matrix.shape[1]):
        if np.sum(feature_matrix[:, i]) == 0:
            CV.append(np.nan)
        else:
            CV.append(variation(feature_matrix[:, i]))

    if np.sum(np.nan_to_num(CV)) == 0:
        return 0
    else:
        return np.nanmean(CV)


def cell_uniformity_fraction(feature_matrix):
    if np.sum(feature_matrix) == 0 or feature_matrix.shape[0] == 1:
        return 1
    else:
        pca = PCA(n_components=1)
        model = pca.fit(feature_matrix)
        fraction = model.explained_variance_ratio_[0]
        return fraction


def weighted_by_cluster(vector, labels):
    for i in range(len(vector)):
        vector[i] = vector[i] * len(np.where(labels == i)[0])
    weighted_average = np.sum(vector) / len(labels)
    return weighted_average


def cell_size_uniformity(mask):
    cell_coord = get_indices_sparse(mask)[1:]
    cell_coord_num = len(cell_coord)
    cell_size = []
    for i in range(cell_coord_num):
        cell_size_current = len(cell_coord[i][0])
        if cell_size_current != 0:
            cell_size.append(cell_size_current)
    cell_size_std = np.std(np.expand_dims(np.array(cell_size), 1))
    return cell_size_std


def cell_type(mask, img, bestz):
    label_list = []
    cell_coord = get_indices_sparse(mask)[1:]
    cell_coord_num = len(cell_coord)
    ss = StandardScaler()
    feature_matrix_z_pieces = []
    for ch_idx, z_idx, channel in img.get_img_channel_generator(z=bestz[0]):
        channel_z = ss.fit_transform(channel[0,0])
        print(f"channel_z is {channel_z.shape} {channel_z.dtype}")
        cell_intensity_z = []
        for j in range(cell_coord_num):
            cell_size_current = len(cell_coord[j][0])
            if cell_size_current != 0:
                single_cell_intensity_z = (
                    np.sum(channel_z[tuple(cell_coord[j])]) / cell_size_current
                )
                cell_intensity_z.append(single_cell_intensity_z)
        feature_matrix_z_pieces.append(cell_intensity_z)

    feature_matrix_z = np.vstack(feature_matrix_z_pieces).T
    for c in range(1, 11):
        model = KMeans(n_clusters=c).fit(feature_matrix_z)
        label_list.append(model.labels_.astype(int))
    return label_list


def cell_uniformity(mask, img, bestz, label_list):
    cell_coord = get_indices_sparse(mask)[1:]
    cell_coord_num = len(cell_coord)
    ss = StandardScaler()
    feature_matrix_pieces = []
    feature_matrix_z_pieces = []
    for ch_idx, z_idx, channel in img.get_img_channel_generator(z=bestz[0]):
        channel_z = ss.fit_transform(channel[0, 0])
        cell_intensity = []
        cell_intensity_z = []
        for j in range(cell_coord_num):
            cell_size_current = len(cell_coord[j][0])
            if cell_size_current != 0:
                single_cell_intensity = np.sum(channel[0,0][tuple(cell_coord[j])]) / cell_size_current
                single_cell_intensity_z = (
                    np.sum(channel_z[tuple(cell_coord[j])]) / cell_size_current
                )
                cell_intensity.append(single_cell_intensity)
                cell_intensity_z.append(single_cell_intensity_z)
        feature_matrix_pieces.append(cell_intensity)
        feature_matrix_z_pieces.append(cell_intensity_z)

    feature_matrix = np.vstack(feature_matrix_pieces).T
    feature_matrix_z = np.vstack(feature_matrix_z_pieces).T
    print(f"cell_uniformity: feature_matrix {feature_matrix.shape} {feature_matrix.dtype}")
    print(f"cell_uniformity: feature_matrix_z {feature_matrix_z.shape} {feature_matrix_z.dtype}")
    CV = []
    fraction = []
    silhouette = []

    for c in range(1, 11):
        labels = label_list[c - 1]
        CV_current = []
        fraction_current = []
        if c == 1:
            silhouette.append(1)
        else:
            silhouette.append(silhouette_score(feature_matrix_z, labels))
        for i in range(c):
            cluster_feature_matrix = feature_matrix[np.where(labels == i)[0], :]
            cluster_feature_matrix_z = feature_matrix_z[np.where(labels == i)[0], :]
            CV_current.append(cell_uniformity_CV(cluster_feature_matrix))
            fraction_current.append(cell_uniformity_fraction(cluster_feature_matrix_z))
        CV.append(weighted_by_cluster(CV_current, labels))
        fraction.append(weighted_by_cluster(fraction_current, labels))
    return CV, fraction, silhouette[1:]


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(np.int64(data.max() + 1), data.size))


def get_indices_sparse(data):
    data = data.astype(np.uint64)
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def get_indexed_mask(mask, boundary):
    boundary = boundary * 1
    boundary_loc = np.where(boundary == 1)
    boundary[boundary_loc] = mask[boundary_loc]
    return boundary


def flatten_dict(input_dict):
    local_list = []
    for key, value in input_dict.items():
        if type(value) == dict:
            local_list.extend(flatten_dict(value))
        else:
            local_list.append(value)
    return local_list


def get_schema_url(ome_xml_root_node: ET.Element) -> str:
    if m := schema_url_pattern.match(ome_xml_root_node.tag):
        return m.group(1)
    raise ValueError(f"Couldn't extract schema URL from tag name {ome_xml_root_node.tag}")


def get_physical_dimension_func(
    dimensions: Literal[2, 3],
) -> Callable[[aicsimageio.AICSImage], Tuple[UnitRegistry, Quantity]]:
    dimension_names = "XYZ"

    def physical_dimension_func(img: aicsimageio.AICSImage) -> Tuple[UnitRegistry, Quantity]:
        """
        Returns area of each pixel (if dimensions == 2) or volume of each
        voxel (if dimensions == 3) as a pint.Quantity. Also returns the
        unit registry associated with these dimensions, with a 'cell' unit
        added to the defaults
        """
        reg = UnitRegistry()
        reg.define("cell = []")

        # aicsimageio parses the OME-XML metadata when loading an image,
        # and uses that metadata to populate various data structures in
        # the AICSImage object. The AICSImage.metadata.to_xml() function
        # constructs a new OME-XML string from that metadata, so anything
        # ignored by aicsimageio won't be present in that XML document.
        # Unfortunately, current aicsimageio ignores physical size units,
        # so we have to parse the original XML ourselves:
        root = ET.fromstring(img.xarray_dask_data.unprocessed[270])
        schema_url = get_schema_url(root)
        pixel_node_attrib = root.findall(f".//{{{schema_url}}}Pixels")[0].attrib

        sizes: List[Quantity] = []
        for _, dimension in zip(range(dimensions), dimension_names):
            # unit = reg[pixel_node_attrib[f"PhysicalSize{dimension}Unit"]]
            unit = reg[pixel_node_attrib.get(f"PhysicalSize{dimension}Unit", "um")]
            # value = float(pixel_node_attrib[f"PhysicalSize{dimension}"])
            value = float(pixel_node_attrib[f"PhysicalSize{dimension}"])
            sizes.append(value * unit)

        size: Quantity = prod(sizes)
        return reg, size

    return physical_dimension_func


get_pixel_area = get_physical_dimension_func(2)


def get_quality_score(features, model):
    ss = model[0]
    pca = model[1]
    features_scaled = ss.transform(features)
    score = (
        pca.transform(features_scaled)[0, 0] * pca.explained_variance_ratio_[0]
        + pca.transform(features_scaled)[0, 1] * pca.explained_variance_ratio_[1]
    )
    return score


def single_method_eval(img, mask, output_dir: Path) -> Tuple[Dict[str, Any], float, float]:
    print("Calculating single-method metrics v1.5 for", img.path)
    # get best z slice for future use
    bestz = mask.bestz
    assert isinstance(bestz, list), "bestz is not a list?"
    if img.data is None:
        if len(bestz) > 1:
            raise RuntimeError("Only a single bestz is currently"
                               " suppored in min-memory mode")
        if img.img.dims.Z != 1:
            raise RuntimeError("Only a single slice is currently"
                               " suppoerted in min-memory mode")

    # get compartment masks
    matched_mask = np.squeeze(mask.data[0, 0, :, bestz, :, :], axis=0)
    np.save(output_dir / "test_matched_mask.npy", matched_mask)
    cell_matched_mask = matched_mask[0]
    nuclear_matched_mask = matched_mask[1]
    cell_outside_nucleus_mask = cell_matched_mask - nuclear_matched_mask
    np.save(output_dir / "test_cell_outside_nucleus_mask.npy", cell_outside_nucleus_mask)

    metric_mask = np.expand_dims(cell_matched_mask, 0)
    metric_mask = np.vstack((metric_mask, np.expand_dims(nuclear_matched_mask, 0)))
    metric_mask = np.vstack((metric_mask, np.expand_dims(cell_outside_nucleus_mask, 0)))
    np.save(output_dir / "test_metric_mask.npy", metric_mask)

    # separate image foreground background
    print("single method point 1")
    pprint(threadpool_info())
    try:
        img_xmldict = xmltodict.parse(img.img.metadata.to_xml())
        seg_channel_names = img_xmldict["OME"]["StructuredAnnotations"]["XMLAnnotation"]["Value"][
            "OriginalMetadata"
        ]["Value"]
        all_channel_names = img.get_channel_labels()
        nuclear_channel_index = all_channel_names.index(seg_channel_names["Nucleus"])
        cell_channel_index = all_channel_names.index(seg_channel_names["Cell"])
        thresholding_channels = [nuclear_channel_index, cell_channel_index]
        seg_channel_provided = True
    except:
        print("single method point 1b")
        thresholding_channels = range(len(img.get_channel_labels()))
        seg_channel_provided = False
    print("single method point 2")
    img_thresholded = np.zeros_like(img.get_plane(0, bestz[0]))
    for c in thresholding_channels:
        img_thresholded += thresholding(img.get_plane(c, bestz[0]))
    print("single method point 3")
    if not seg_channel_provided:
        thresh_lim = round(len(img.get_channel_labels()) * 0.2)
        img_thresholded[img_thresholded <= thresh_lim] = 0
    img_thresholded = img_thresholded[0, :, :]
    np.save(output_dir / "test_img_thresholded.npy", img_thresholded)
    img_binary = foreground_separation(img_thresholded)
    img_binary = np.sign(img_binary)
    print(f"img_binary {img_binary.shape} {img_binary.dtype}")
    np.save(output_dir / "test_img_binary.npy", img_binary)
    background_pixel_num = np.argwhere(img_binary == 0).shape[0]
    fraction_background = background_pixel_num / (img_binary.shape[0] * img_binary.shape[1])
    # np.savetxt(output_dir / f"{img.name}_img_binary.txt.gz", img_binary)
    fg_bg_image = Image.fromarray(img_binary.astype(np.uint8) * 255, mode="L").convert("1")
    fg_bg_image.save(output_dir / f"{img.name}_img_binary.png")

    print(f"single method point 4; metric_mask {metric_mask.shape} {metric_mask.dtype}")
    # set mask channel names
    channel_names = [
        "Matched Cell",
        "Nucleus (including nuclear membrane)",
        "Cell Not Including Nucleus (cell membrane plus cytoplasm)",
    ]
    metrics = {}
    for channel in range(metric_mask.shape[0]):
        print(f"single method point 5 {channel} of {metric_mask.shape[0]}")
        current_mask = metric_mask[channel]
        mask_binary = np.sign(current_mask)
        metrics[channel_names[channel]] = {}
        if channel_names[channel] == "Matched Cell":
            print(f"single method point 5 case 1 {channel_names[channel]} {mask_binary.dtype}")
            mask_xmldict = xmltodict.parse(mask.img.metadata.to_xml())
            try:
                matched_fraction = mask_xmldict["OME"]["StructuredAnnotations"]["XMLAnnotation"][
                    "Value"
                ]["OriginalMetadata"]["Value"]
            except:
                matched_fraction = 1.0

            units, pixel_size = get_pixel_area(img.img)
            print(f"units {units} pixel_size {pixel_size}")

            pixel_num = mask_binary.shape[0] * mask_binary.shape[1]
            total_area = pixel_size * pixel_num

            # calculate number of cell per 100 squared micron
            cell_num = units["cell"] * len(np.unique(current_mask)) - 1

            cells_per_area = cell_num / total_area
            units.define("hundred_square_micron = micrometer ** 2 * 100")
            cell_num_normalized = cells_per_area.to("cell / hundred_square_micron")

            # calculate the standard deviation of cell size

            cell_size_std = cell_size_uniformity(current_mask)

            # get coverage metrics
            foreground_fraction, background_fraction, mask_foreground_fraction = fraction(
                img_binary, mask_binary
            )

            foreground_CV, foreground_PCA = foreground_uniformity(
                img_binary, mask_binary, img, bestz
            )
            background_CV, background_PCA = background_uniformity(img_binary, img, bestz)
            metrics[channel_names[channel]][
                "NumberOfCellsPer100SquareMicrons"
            ] = cell_num_normalized.magnitude
            metrics[channel_names[channel]][
                "FractionOfForegroundOccupiedByCells"
            ] = foreground_fraction
            metrics[channel_names[channel]]["1-FractionOfBackgroundOccupiedByCells"] = (
                1 - background_fraction
            )
            metrics[channel_names[channel]][
                "FractionOfCellMaskInForeground"
            ] = mask_foreground_fraction
            metrics[channel_names[channel]]["1/(ln(StandardDeviationOfCellSize)+1)"] = 1 / (
                np.log(cell_size_std) + 1
            )
            metrics[channel_names[channel]]["FractionOfMatchedCellsAndNuclei"] = matched_fraction
            metrics[channel_names[channel]]["1/(AvgCVForegroundOutsideCells+1)"] = 1 / (
                foreground_CV + 1
            )
            metrics[channel_names[channel]][
                "FractionOfFirstPCForegroundOutsideCells"
            ] = foreground_PCA

            cell_type_labels = cell_type(current_mask, img, bestz)
        else:
            print(f"single method point 5 case 2 {channel_names[channel]}")
            #img_channels = np.squeeze(img.data[0, 0, :, bestz, :, :], axis=0)
            # get cell uniformity
            cell_CV, cell_fraction, cell_silhouette = cell_uniformity(
                current_mask, img, bestz, cell_type_labels
            )
            avg_cell_CV = np.average(cell_CV)
            avg_cell_fraction = np.average(cell_fraction)
            avg_cell_silhouette = np.average(cell_silhouette)

            metrics[channel_names[channel]][
                "1/(AvgOfWeightedAvgCVMeanCellIntensitiesOver1~10NumberOfClusters+1)"
            ] = 1 / (avg_cell_CV + 1)
            metrics[channel_names[channel]][
                "AvgOfWeightedAvgFractionOfFirstPCMeanCellIntensitiesOver1~10NumberOfClusters"
            ] = avg_cell_fraction
            metrics[channel_names[channel]][
                "AvgSilhouetteOver2~10NumberOfClusters"
            ] = avg_cell_silhouette

    print("single method point 6")
    metrics_flat = np.expand_dims(flatten_dict(metrics), 0)
    with importlib.resources.open_binary("sprm", "pca.pickle") as f:
        PCA_model = pickle.load(f)

    quality_score = get_quality_score(metrics_flat, PCA_model)
    metrics["QualityScore"] = quality_score

    print("single method point 7")
    return metrics, fraction_background, 1 / (background_CV + 1), background_PCA
