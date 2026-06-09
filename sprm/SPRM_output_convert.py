#!/usr/bin/env python3
import json
from argparse import ArgumentParser
from collections import defaultdict
from math import ceil, log2
from os import walk
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional, Union
from xml.etree import ElementTree

import anndata
import numpy as np
import pandas as pd
import spatialdata
import tifffile
from bioio import BioImage
from scipy.io import mmread
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    TableModel,
)

desired_pixel_size_for_pyramid = 250


def load_adjacency_matrix_and_labels(
    adjacency_file: Path, label_file: Path, adata: anndata.AnnData
):
    adjacency_matrix = mmread(adjacency_file).tocsc()
    labels = pd.read_csv(label_file, header=None, names=["cell_id"], sep=r"\s+")

    adata_cell_ids = adata.obs.index.astype(int).to_list()
    filtered_labels = labels[labels["cell_id"].isin(adata_cell_ids)]
    filtered_cell_ids = filtered_labels["cell_id"].values

    label_to_index_map = pd.Series(labels.index.values, index=labels["cell_id"].astype(int))
    filtered_indices = label_to_index_map[filtered_cell_ids].values

    # Adjust indices to fit the zero-based indexing
    adjusted_indices = filtered_indices - 1
    filtered_matrix = adjacency_matrix[adjusted_indices, :][:, adjusted_indices]
    return filtered_matrix


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if "[" not in column and " " not in column:
            continue
        sanitized_column = column.replace("[", "").replace("]", "").replace(" ", "-")
        df[sanitized_column] = df[column]
        df = df.drop(column, axis=1, inplace=False)
    return df


def find_ome_tiffs(directory: Path) -> Iterable[Path]:
    yield from find_files(directory, "*.ome.tif*")


def find_files(directory: Path, pattern: str) -> Iterable[Path]:
    for dirpath_str, dirnames, filenames in walk(directory):
        dirpath = Path(dirpath_str)
        for filename in filenames:
            filepath = dirpath / filename
            if filepath.match(pattern):
                yield filepath


def read_expr_img(
    expr_img_path: Path, nmf: bool = False
) -> tuple[Union[Image2DModel, Image3DModel], tuple[int, ...]]:
    image = BioImage(expr_img_path)
    image_data_squeezed = image.data.squeeze()
    image_scale_factors = (2,) * ceil(
        log2(max(image_data_squeezed.shape[1:]) / desired_pixel_size_for_pyramid)
    )
    is_3d = image.dims.Z > 1
    model = Image3DModel if is_3d else Image2DModel
    if nmf:
        return (
            model.parse(
                image_data_squeezed, scale_factors=image_scale_factors, dims=["y", "x", "c"]
            ),
            image_scale_factors,
        )
    else:
        return (
            model.parse(
                image_data_squeezed,
                c_coords=image.channel_names,
                scale_factors=image_scale_factors,
            ),
            image_scale_factors,
        )


def read_mask_img(
    mask_path: Path, scale_factors: tuple[int, ...], superpixel: bool = False
) -> dict[str, Union[Labels2DModel, Labels3DModel]]:
    image = BioImage(mask_path)
    mask_channel_names = image.channel_names
    mask_img_arr = tifffile.imread(mask_path)
    is_3d = image.dims.Z > 1
    model = Labels3DModel if is_3d else Labels2DModel
    mask_dict = {}
    if superpixel:
        mask_dict["superpixel"] = model.parse(
            data=image.data.squeeze(), scale_factors=scale_factors
        )
    else:
        for i, ch in enumerate(mask_channel_names):
            mask_dict[ch] = model.parse(data=mask_img_arr[i], scale_factors=scale_factors)
    return mask_dict


def read_table(sprm_dir: Path, expr_img_path: Path, spatialdata_dir: Path) -> TableModel:
    mean_csv = sprm_dir / Path(f"{expr_img_path.name}-cell_channel_mean.csv")
    mean_df = pd.read_csv(mean_csv)
    features = list(mean_df.columns)[1:]
    observations = [int(i) for i in list(mean_df.ID)]
    var = pd.DataFrame(index=features)
    obs = pd.DataFrame(index=observations)
    x = mean_df.drop("ID", axis=1, inplace=False).to_numpy()
    adata = anndata.AnnData(X=x, obs=obs, var=var)

    adata.obs.index = [int(i) for i in adata.obs.index]
    adata.obs["cell_id"] = adata.obs.index
    adata.obs["region"] = "cells"
    adata.uns["spatialdata_attrs"] = {
        "instance_key": "cell_id",
        "region": "cells",
        "region_key": "region",
    }

    total_csv = sprm_dir / Path(f"{expr_img_path.name}-cell_channel_total.csv")
    total_df = pd.read_csv(total_csv)
    adata.layers["total"] = total_df.drop("ID", axis=1, inplace=False).to_numpy()

    cluster_csv = sprm_dir / Path(f"{expr_img_path.name}-cell_cluster.csv")

    cluster_df = pd.read_csv(cluster_csv).set_index("ID", drop=True, inplace=False)
    for column in cluster_df.columns:
        adata.obs[column] = cluster_df[column]

    adjacency_matrix_path = sprm_dir / Path(f"{expr_img_path.name}_AdjacencyMatrix.mtx")

    adjacency_matrix_labels_path = sprm_dir / Path(
        f"{expr_img_path.name}_AdjacencyMatrixRowColLabels.txt"
    )

    adjacency_matrix = load_adjacency_matrix_and_labels(
        adjacency_matrix_path, adjacency_matrix_labels_path, adata
    )
    adata.obsp["adjacency_matrix"] = adjacency_matrix

    tsne_csv = sprm_dir / Path(f"{expr_img_path.name}-tSNE_allfeatures.csv")

    tsne_df = pd.read_csv(tsne_csv)
    tsne_coords = tsne_df.drop("ID", axis=1, inplace=False).to_numpy()
    adata.obsm["tSNE"] = tsne_coords

    covariance_matrix_paths = [
        f"{expr_img_path.name}-cell_channel_covar.npy",
        f"{expr_img_path.name}-nuclei_channel_covar.npy",
        f"{expr_img_path.name}-cell_boundaries_channel_covar.npy",
        f"{expr_img_path.name}-nucleus_boundaries_channel_covar.npy",
    ]
    for covariance_matrix_path in covariance_matrix_paths:
        a = np.ndarray((len(adata.var.index), len(adata.var.index), len(adata.obs.index)))
        matrix = np.load(spatialdata_dir / covariance_matrix_path)
        a = matrix.reshape(
            (len(adata.obs.index), len(adata.var.index), len(adata.var.index))
        ).transpose((1, 2, 0))
        adata.varp[
            covariance_matrix_path.replace(f"{expr_img_path.name}-", "").replace(".npy", "")
        ] = a

    adata.obs = sanitize_column_names(adata.obs)
    for column in adata.obs.columns:
        if column not in {"cell_id"}:
            adata.obs[column] = adata.obs[column].astype("category")
    return TableModel.parse(adata)


def get_segmentation_channels(ome_tiff: Path) -> dict[str, list[str]]:
    ome_xml = tifffile.tiffcomment(ome_tiff)
    t = ElementTree.fromstring(ome_xml)
    ome_namespace = t.tag.split("}")[0].lstrip("{")
    nsmap = {"OME": ome_namespace}
    xpath_pieces = [
        "OME:StructuredAnnotations",
        "OME:XMLAnnotation",
        "OME:Value",
        "OME:OriginalMetadata",
        "OME:Key[.='SegmentationChannels']",
        "..",
        "OME:Value",
    ]
    annotation_node = t.find("/".join(xpath_pieces), nsmap)
    metadata = defaultdict(list)
    for ch_node in annotation_node:
        tag = ch_node.tag.split("}")[1]
        key = f"{tag}SegmentationChannels"
        metadata[key].append(ch_node.text.strip())
    return dict(metadata)


def get_sprm_qc_measures(sprm_dir: Path, image_name: str):
    sprm_qc_json = sprm_dir / f"{image_name}-SPRM_Image_Quality_Measures.json"
    with open(sprm_qc_json) as f:
        qc_data = json.load(f)
    zscore_parent = qc_data["Image Quality Metrics not requiring image segmentation"]
    mean_snz = mean(zscore_parent["Signal To Noise Z-Score"].values())
    seg_eval_data = qc_data["Segmentation Evaluation Metrics"]
    mc_data = seg_eval_data.get("Matched Cell")
    acvf = mc_data.get("1/(AvgCVForegroundOutsideCells+1)")
    quality_score = seg_eval_data["QualityScore"]
    return {
        "QualityScore": quality_score,
        "Mean_SNZ": mean_snz,
        "ACVF": acvf,
    }


def write_segmentation_metadata_json(img_dir: Path, sprm_dir: Path):
    channel_metadata = {
        image.name: get_segmentation_channels(image) for image in img_dir.glob("*.ome.tiff")
    }
    sprm_qc = {image: get_sprm_qc_measures(sprm_dir, image) for image in channel_metadata}

    metadata_reshaped = []
    for image, channels in channel_metadata.items():
        image_metadata = {"Image": image}
        image_metadata |= channels
        image_metadata |= sprm_qc[image]
        metadata_reshaped.append(image_metadata)

    with open("segmentation-metadata.json", "w") as f:
        json.dump(metadata_reshaped, f)


def main(
    img_dir: Path,
    mask_dir: Path,
    sprm_dir: Path,
    spatialdata_dir: Optional[Path] = None,
):
    write_segmentation_metadata_json(img_dir, sprm_dir)
    expr_image_paths = find_ome_tiffs(img_dir)
    for expr_image in expr_image_paths:
        expr_img, scale_factors = read_expr_img(expr_image)
        mask_path = mask_dir / expr_image.name.replace("expr", "mask")
        mask_img_dict = read_mask_img(mask_path, scale_factors)
        table = read_table(sprm_dir, expr_image, spatialdata_dir)
        images_dict = {"expr": expr_img}

        pca_path = sprm_dir / (expr_image.name + "-channel_pca.ome.tiff")
        if pca_path.exists():
            print("path exists")
            pca_img, scale_factors = read_expr_img(pca_path)
            images_dict["pca"] = pca_img
        super_pixel_path = sprm_dir / (expr_image.name + "-superpixel.ome.tiff")
        if super_pixel_path.exists():
            print("superpixel path exists")
            super_pixel_mask = read_mask_img(super_pixel_path, scale_factors, superpixel=True)
            mask_img_dict.update(super_pixel_mask)
        nmf_path = sprm_dir / (expr_image.name + "_nmf_top3.png")
        if nmf_path.exists():
            print("path exists")
            nmf_img, scale_factors = read_expr_img(nmf_path, nmf=True)
            images_dict["nmf"] = nmf_img

        #        print(mask_img_dict)
        sdata = spatialdata.SpatialData(
            images=images_dict,
            labels=mask_img_dict,
            tables={"table": table},
        )
        sdata.write(f"{expr_image.stem}_sprm_output.zarr")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--sprm-dir", type=Path, required=True)
    p.add_argument("--spatialdata-dir", type=Path, required=False)
    args = p.parse_args()

    main(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        sprm_dir=args.sprm_dir,
        spatialdata_dir=args.spatialdata_dir,
    )
