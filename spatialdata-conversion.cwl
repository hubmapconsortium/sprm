#!/usr/bin/env cwl-runner
class: Workflow
cwlVersion: v1.1
label: SPRM pipeline

inputs:
  image_dir:
    label: "Directory containing image files"
    type: Directory
  mask_dir:
    label: "Directory containing mask files"
    type: Directory
  sprm_dir:
    label: "SPRM output directory"
    type: Directory
  num_dims:
    label: "Number of dimensions associated with the data"
    type: int?

outputs:
  sdata_zarr:
    outputSource: create_spatial_data/sdata_zarr
    type: Directory

steps:
  create_spatial_data:
    in:
      image_dir:
        source: ome_tiff_normalize_expr/output_dir
      mask_dir:
        source: ome_tiff_normalize_mask/output_dir
      sprm_dir:
        source: sprm/output_dir
      num_dims:
        source: num_dims
    out: [sdata_zarr]
    run: steps/create-spatial-data.cwl
    label: "Conversion to spatialdata format"