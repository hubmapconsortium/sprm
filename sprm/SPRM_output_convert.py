import anndata
import pandas as pd
import spatialdata
import tifffile
from argparse import ArgumentParser
from pathlib import Path
from typing import Union, Dict, Tuple
from scipy.io import mmread
from os import walk

from spatialdata.models import Image2DModel, Image3DModel, Labels2DModel, Labels3DModel, PointsModel, TableModel
from aicsimageio import AICSImage

from .SPRM_pkg import *
from math import ceil, log2


desired_pixel_size_for_pyramid = 250

def find_ome_tiff(directory: Path)->Path:
    return find_file(directory, "*.ome.tif*")

def find_file(directory: Path, pattern:str) -> Path:
    for dirpath_str, dirnames, filenames in walk(directory):
        dirpath = Path(dirpath_str)
        for filename in filenames:
            filepath = dirpath / filename
            if filepath.match(pattern):
                return filepath

def read_expr_img(img_dir:Path, num_dims:int)->Tuple[Union[Image2DModel, Image3DModel], Tuple]:

    model_dict = {2:Image2DModel, 3:Image3DModel}
    expr_img_path = find_ome_tiff(img_dir)
    image = AICSImage(expr_img_path)
    image_data_squeezed = image.data.squeeze()
    image_scale_factors = (2,) * ceil(
        log2(max(image_data_squeezed.shape[1:]) / desired_pixel_size_for_pyramid)
    )
    model = model_dict[num_dims]
    return model.parse(image_data_squeezed, c_coords=image.channel_names, scale_factors=image_scale_factors), image_scale_factors

def read_mask_img(mask_dir:Path, num_dims:int, scale_factors: Tuple)->Dict[str, Union[Labels2DModel, Labels3DModel]]:
    model_dict = {2:Labels2DModel, 3:Labels3DModel}
    mask_img_path = find_ome_tiff(mask_dir)
    aicsimage = AICSImage(mask_img_path)
    mask_channel_names = aicsimage.channel_names
    mask_img_arr = tifffile.imread(mask_img_path)
    model = model_dict[num_dims]
    mask_dict = {}
    for i in range(len(mask_channel_names)):
        mask_dict[mask_channel_names[i]] = model.parse(data=mask_img_arr[i], scale_factors=scale_factors)
    return mask_dict

def read_table(sprm_dir)->TableModel:
    mean_csv = find_file(sprm_dir, '*cell_channel_mean.csv')
    mean_df = pd.read_csv(mean_csv)
    features = list(mean_df.columns)[1:]
    observations = list(mean_df.ID)
    var = pd.DataFrame(index=features)
    obs = pd.DataFrame(index=observations)
    x = mean_df.drop('ID', inplace=False).to_numpy()
    adata = anndata.AnnData(X=x, obs=obs, var=var)

    total_csv = find_file(sprm_dir, '*cell_channel_total.csv')
    total_df = pd.read_csv(total_csv)
    adata.layers["total"] = total_df.drop("ID", inplace=False).to_numpy()
    
    cluster_csv = find_file(sprm_dir, '*cell_cluster.csv')
    cluster_df = pd.read_csv(cluster_csv).set_index('ID',drop=True, inplace=False)
    for column in cluster_df.columns:
        adata.obs[column] = cluster_df[column]

    adjacency_matrix_path = find_file(sprm_dir, "*AdjacencyMatrix.mtx")
    adjacency_matrix = mmread(adjacency_matrix_path)
    adata.obsp['adjacency_matrix'] = adjacency_matrix

    tsne_csv = find_file(sprm_dir, "*tSNE_allfeatures.csv")
    tsne_df = pd.read_csv(tsne_csv)
    tsne_coords = tsne_df.drop('ID', inplace=False).to_numpy()
    adata.obsm["tSNE"] = tsne_coords

    return TableModel.parse(adata)

def main(
    img_dir: Path,
    mask_dir: Path,
    sprm_dir: Path,
    num_dims: int,
):
    expr_img, scale_factors = read_expr_img(img_dir, num_dims)
    mask_img_dict = read_mask_img(mask_dir, num_dims, scale_factors)
    table = read_table(sprm_dir)

    sdata = spatialdata.SpatialData(images={"expr":expr_img}, labels=mask_img_dict, table=table)
    sdata.write_zarr("sprm_output.zarr")


p = ArgumentParser()
p.add_argument("--img-dir", type=Path, required=True)
p.add_argument("--mask-dir", type=Path, required=True)
p.add_argument("--sprm-dir", type=Path, required=True)
p.add_argument("--num-dims", type=int, required=True)

args = p.parse_args()


main(
    img_dir=args.img_dir,
    mask_dir=args.mask_dir,
    sprm_dir=args.sprm_dir,

)
