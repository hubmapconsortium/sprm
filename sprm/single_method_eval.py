import importlib.resources
import pickle
import re
import xml.etree.ElementTree as ET
from math import prod
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

"""
Companion to SPRM.py
Package functions that evaluate a single segmentation method
Author: Haoran Chen and Ted Zhang
Version: 1.5
09/08/2021
"""

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
    contour_ref = img_thre.copy()
    img_thre = closing(img_thre, disk(1))

    img_thre = -img_thre + 1
    img_thre = closing(img_thre, disk(2))
    img_thre = -img_thre + 1

    img_thre = closing(img_thre, disk(20))

    img_thre = -img_thre + 1
    img_thre = closing(img_thre, disk(10))
    img_thre = -img_thre + 1

    img_thre = area_closing(img_thre, 20000, connectivity=2)
    contour_ref = contour_ref.astype(float)
    img_thre = img_thre.astype(float)
    img_binary = MorphGAC(
        -contour_ref + 1, 5, -img_thre + 1, smoothing=1, balloon=0.8, threshold=0.5
    )
    img_binary = area_closing(img_binary, 1000, connectivity=2)

    return -img_binary + 1


def uniformity_CV(loc, channels):
    CV = []
    n = len(channels)
    for i in range(n):
        channel = channels[i]
        channel = channel / np.mean(channel)
        intensity = channel[tuple(loc.T)]
        CV.append(np.std(intensity))
    return np.average(CV)


def uniformity_fraction(loc, channels) -> float:
    n = len(channels)
    feature_matrix_pieces = []
    for i in range(n):
        channel = channels[i]
        ss = StandardScaler()
        channel_z = ss.fit_transform(channel.copy())
        intensity = channel_z[tuple(loc.T)]
        feature_matrix_pieces.append(intensity)
    feature_matrix = np.vstack(feature_matrix_pieces)
    pca = PCA(n_components=1)
    model = pca.fit(feature_matrix.T)
    fraction = model.explained_variance_ratio_[0]
    return fraction


def foreground_uniformity(img_bi, mask, channels):
    foreground_loc = np.argwhere((img_bi - mask) == 1)
    CV = uniformity_CV(foreground_loc, channels)
    fraction = uniformity_fraction(foreground_loc, channels)
    return CV, fraction


def background_uniformity(img_bi, channels):
    background_loc = np.argwhere(img_bi == 0)
    CV = uniformity_CV(background_loc, channels)
    background_pixel_num = background_loc.shape[0]
    background_loc_fraction = 1
    while background_loc_fraction > 0:
        try:
            background_loc_sampled = background_loc[
                np.random.randint(
                    background_pixel_num,
                    size=round(background_pixel_num * background_loc_fraction),
                ),
                :,
            ]
            fraction = uniformity_fraction(background_loc_sampled, channels)
            break
        except:
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


def cell_type(mask, channels):
    label_list = []
    n = len(channels)
    cell_coord = get_indices_sparse(mask)[1:]
    cell_coord_num = len(cell_coord)
    ss = StandardScaler()
    feature_matrix_z_pieces = []
    for i in range(n):
        channel = channels[i]
        channel_z = ss.fit_transform(channel)
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


def cell_uniformity(mask, channels, label_list):
    n = len(channels)
    cell_coord = get_indices_sparse(mask)[1:]
    cell_coord_num = len(cell_coord)
    ss = StandardScaler()
    feature_matrix_pieces = []
    feature_matrix_z_pieces = []
    for i in range(n):
        channel = channels[i]
        channel_z = ss.fit_transform(channel)
        cell_intensity = []
        cell_intensity_z = []
        for j in range(cell_coord_num):
            cell_size_current = len(cell_coord[j][0])
            if cell_size_current != 0:
                single_cell_intensity = np.sum(channel[tuple(cell_coord[j])]) / cell_size_current
                single_cell_intensity_z = (
                    np.sum(channel_z[tuple(cell_coord[j])]) / cell_size_current
                )
                cell_intensity.append(single_cell_intensity)
                cell_intensity_z.append(single_cell_intensity_z)
        feature_matrix_pieces.append(cell_intensity)
        feature_matrix_z_pieces.append(cell_intensity_z)

    feature_matrix = np.vstack(feature_matrix_pieces).T
    feature_matrix_z = np.vstack(feature_matrix_z_pieces).T
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
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
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


def get_pixel_area(pixel_node_attrib: Dict[str, str]) -> float:
    """
    Returns total pixel size in square micrometers.
    """
    reg = UnitRegistry()

    sizes: List[Quantity] = []
    for dimension in ["X", "Y"]:
        unit = reg[pixel_node_attrib[f"PhysicalSize{dimension}Unit"]]
        value = float(pixel_node_attrib[f"PhysicalSize{dimension}"])
        sizes.append(value * unit)

    size = prod(sizes)
    return size.to("micrometer ** 2").magnitude


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

    # get compartment masks
    matched_mask = np.squeeze(mask.data[0, 0, :, bestz, :, :], axis=0)
    cell_matched_mask = matched_mask[0]
    nuclear_matched_mask = matched_mask[1]
    cell_outside_nucleus_mask = cell_matched_mask - nuclear_matched_mask

    metric_mask = np.expand_dims(cell_matched_mask, 0)
    metric_mask = np.vstack((metric_mask, np.expand_dims(nuclear_matched_mask, 0)))
    metric_mask = np.vstack((metric_mask, np.expand_dims(cell_outside_nucleus_mask, 0)))

    # separate image foreground background
    try:
        img_xmldict = xmltodict.parse(img.img.metadata.to_xml())
        seg_channel_names = img_xmldict["OME"]["StructuredAnnotations"]["XMLAnnotation"]["Value"][
            "OriginalMetadata"
        ]["Value"]
        all_channel_names = img.img.get_channel_names()
        nuclear_channel_index = all_channel_names.index(seg_channel_names["Nucleus"])
        cell_channel_index = all_channel_names.index(seg_channel_names["Cell"])
        thresholding_channels = [nuclear_channel_index, cell_channel_index]
        seg_channel_provided = True
    except:
        thresholding_channels = range(img.data.shape[2])
        seg_channel_provided = False
    img_thresholded = sum(
        thresholding(np.squeeze(img.data[0, 0, c, bestz, :, :], axis=0))
        for c in thresholding_channels
    )
    if not seg_channel_provided:
        img_thresholded[img_thresholded <= round(img.data.shape[2] * 0.2)] = 0
    img_binary = foreground_separation(img_thresholded)
    img_binary = np.sign(img_binary)
    background_pixel_num = np.argwhere(img_binary == 0).shape[0]
    fraction_background = background_pixel_num / (img_binary.shape[0] * img_binary.shape[1])
    # np.savetxt(output_dir / f"{img.name}_img_binary.txt.gz", img_binary)
    fg_bg_image = Image.fromarray(img_binary.astype(np.uint8) * 255, mode="L").convert("1")
    fg_bg_image.save(output_dir / f"{img.name}_img_binary.png")

    # set mask channel names
    channel_names = [
        "Matched Cell",
        "Nucleus (including nuclear membrane)",
        "Cell Not Including Nucleus (cell membrane plus cytoplasm)",
    ]
    metrics = {}
    for channel in range(metric_mask.shape[0]):
        current_mask = metric_mask[channel]
        mask_binary = np.sign(current_mask)
        metrics[channel_names[channel]] = {}
        if channel_names[channel] == "Matched Cell":
            mask_xmldict = xmltodict.parse(mask.img.metadata.to_xml())
            try:
                matched_fraction = mask_xmldict["OME"]["StructuredAnnotations"]["XMLAnnotation"][
                    "Value"
                ]["OriginalMetadata"]["Value"]
            except:
                matched_fraction = 1.0

            schema_url = get_schema_url(img.img.metadata.root_node)
            pixels_node = img.img.metadata.dom.findall(f".//{{{schema_url}}}Pixels")[0]
            pixel_size = get_pixel_area(pixels_node.attrib)

            pixel_num = mask_binary.shape[0] * mask_binary.shape[1]
            micron_num = pixel_size * pixel_num

            # calculate number of cell per 100 squared micron
            cell_num = len(np.unique(current_mask)) - 1

            cell_num_normalized = cell_num / micron_num * 100

            # calculate the standard deviation of cell size

            cell_size_std = cell_size_uniformity(current_mask)

            # get coverage metrics
            foreground_fraction, background_fraction, mask_foreground_fraction = fraction(
                img_binary, mask_binary
            )

            img_channels = np.squeeze(img.data[0, 0, :, bestz, :, :], axis=0)

            foreground_CV, foreground_PCA = foreground_uniformity(
                img_binary, mask_binary, img_channels
            )
            background_CV, background_PCA = background_uniformity(img_binary, img_channels)
            metrics[channel_names[channel]][
                "NumberOfCellsPer100SquareMicrons"
            ] = cell_num_normalized
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

            # get cell type labels
            cell_type_labels = cell_type(current_mask, img_channels)
        else:
            img_channels = np.squeeze(img.data[0, 0, :, bestz, :, :], axis=0)
            # get cell uniformity
            cell_CV, cell_fraction, cell_silhouette = cell_uniformity(
                current_mask, img_channels, cell_type_labels
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

    metrics_flat = np.expand_dims(flatten_dict(metrics), 0)
    with importlib.resources.open_binary("sprm", "pca.pickle") as f:
        PCA_model = pickle.load(f)

    quality_score = get_quality_score(metrics_flat, PCA_model)
    metrics["QualityScore"] = quality_score

    return metrics, fraction_background, 1 / (background_CV + 1), background_PCA
