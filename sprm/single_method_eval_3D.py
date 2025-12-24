import importlib.resources
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import xmltodict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .single_method_eval import (
    cell_size_uniformity,
    cell_uniformity_CV,
    cell_uniformity_fraction,
    flatten_dict,
    foreground_separation,
    get_indices_sparse,
    _get_metadata_xml,
    get_physical_dimension_func,
    get_quality_score,
    thresholding,
    uniformity_CV,
    weighted_by_cluster,
)

"""
Companion to SPRM.py
Package functions that evaluate a single segmentation method
Author: Haoran Chen and Ted Zhang
Version: 2.0.1
04/21/2022 - 12/23/2025
"""

get_voxel_volume = get_physical_dimension_func(3)


def fraction(img_bi, mask_bi):
    foreground_all = np.sum(img_bi)
    background_all = img_bi.shape[0] * img_bi.shape[1] * img_bi.shape[2] - foreground_all
    mask_all = np.sum(mask_bi)
    background = len(np.where(mask_bi - img_bi == 1)[0])
    foreground = np.sum(mask_bi * img_bi)
    if background_all == 0:
        background_fraction = 0
    else:
        background_fraction = background / background_all
    foreground_fraction = foreground / foreground_all
    mask_fraction = foreground / mask_all
    return foreground_fraction, background_fraction, mask_fraction


def uniformity_fraction(loc, channels) -> float:
    n = len(channels)
    feature_matrix_pieces = []
    for i in range(n):
        channel = channels[i]
        ss = StandardScaler()
        z, x, y = channel.shape
        channel_z = ss.fit_transform(channel.reshape(z, x * y)).reshape(z, x, y)
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
    fraction = uniformity_fraction(background_loc, channels)
    return CV, fraction


def cell_type(mask, channels):
    label_list = []
    n = len(channels)
    cell_coord = get_indices_sparse(mask)[1:]
    cell_coord_num = len(cell_coord)
    ss = StandardScaler()
    feature_matrix_z_pieces = []
    for i in range(n):
        channel = channels[i]
        z, x, y = channel.shape
        channel_z = ss.fit_transform(channel.reshape(z, x * y)).reshape(z, x, y)
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
        z, x, y = channel.shape
        channel_z = ss.fit_transform(channel.reshape(z, x * y)).reshape(z, x, y)
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


def single_method_eval_3D(img, mask, output_dir: Path) -> Tuple[Dict[str, Any], float, float]:
    if not img.data:
        raise NotImplementedError("Not implemented for disk-based images")
    print("Calculating single-method metrics 3D v1.5 for", img.path)

    # get compartment masks
    matched_mask = mask.data[0, 0, :, :, :, :]
    cell_matched_mask = matched_mask[0]
    nuclear_matched_mask = matched_mask[1]
    cell_outside_nucleus_mask = cell_matched_mask - nuclear_matched_mask

    metric_mask = np.expand_dims(cell_matched_mask, 0)
    metric_mask = np.vstack((metric_mask, np.expand_dims(nuclear_matched_mask, 0)))
    metric_mask = np.vstack((metric_mask, np.expand_dims(cell_outside_nucleus_mask, 0)))

    # separate image foreground background
    try:
        img_xml = _get_metadata_xml(img.img.metadata)
        if img_xml is None:
            raise ValueError("No parseable OME-XML metadata found on image")
        img_xmldict = xmltodict.parse(img_xml)
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
    img_thresholded = sum(thresholding(img.data[0, 0, c, :, :, :]) for c in thresholding_channels)
    if not seg_channel_provided:
        img_thresholded[img_thresholded <= round(img.data.shape[2] * 0.2)] = 0
    img_binary_pieces = []
    for z in range(img_thresholded.shape[0]):
        img_binary_pieces.append(foreground_separation(img_thresholded[z]))
    img_binary = np.stack(img_binary_pieces, axis=0)
    img_binary = np.sign(img_binary)
    background_voxel_num = np.argwhere(img_binary == 0).shape[0]
    fraction_background = background_voxel_num / (
        img_binary.shape[0] * img_binary.shape[1] * img_binary.shape[2]
    )

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
        img_channels = img.data[0, 0, :, :, :, :]
        if channel_names[channel] == "Matched Cell":
            units, voxel_size = get_voxel_volume(img.img)

            voxel_num = mask_binary.shape[0] * mask_binary.shape[1] * mask_binary.shape[2]
            total_volume = voxel_size * voxel_num

            # TODO: match 3D cell and nuclei and calculate the fraction of match, assume cell and nuclei are matched for now

            # calculate number of cell per 100 cubic micron
            cell_num = units["cell"] * len(np.unique(current_mask)) - 1

            cells_per_volume = cell_num / total_volume
            units.define("hundred_cubic_micron = micrometer ** 3 * 100")
            cell_num_normalized = cells_per_volume.to("cell / hundred_cubic_micron")

            # calculate the standard deviation of cell size
            cell_size_std = cell_size_uniformity(current_mask)

            # get coverage metrics
            foreground_fraction, background_fraction, mask_foreground_fraction = fraction(
                img_binary, mask_binary
            )

            foreground_CV, foreground_PCA = foreground_uniformity(
                img_binary, mask_binary, img_channels
            )
            background_CV, background_PCA = background_uniformity(img_binary, img_channels)
            metrics[channel_names[channel]][
                "NumberOfCellsPer100CubicMicrons"
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
            metrics[channel_names[channel]]["1/(AvgCVForegroundOutsideCells+1)"] = 1 / (
                foreground_CV + 1
            )
            metrics[channel_names[channel]][
                "FractionOfFirstPCForegroundOutsideCells"
            ] = foreground_PCA

            # get cell type labels
            cell_type_labels = cell_type(current_mask, img_channels)
        else:
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
    print(metrics_flat)
    with importlib.resources.open_binary("sprm", "pca_3D.pickle") as f:
        PCA_model = pickle.load(f)

    # generate quality score
    quality_score = get_quality_score(metrics_flat, PCA_model)
    metrics["QualityScore"] = quality_score

    return metrics, fraction_background, 1 / (background_CV + 1), background_PCA
