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
  options_file:
    label: "SPRM options file"
    type: File?
  cell_types_directory:
    label: "Cell type annotations file"
    type: Directory[]?
  options_preset:
    label: "SPRM options preset (alternate options file bundled with the package)"
    type: string?
  enable_manhole:
    label: "Whether to enable remote debugging via 'manhole'"
    type: boolean?
  enable_faulthandler:
    label: "Whether to enable the Python 'faulthandler' module"
    type: boolean?
  verbose:
    label: "Whether to enable verbose/debug mode"
    type: boolean?
  processes:
    label: "Number of images to process in parallel (default: 1)"
    type: int?

outputs:
  sprm_output:
    outputSource: sprm/output_dir
    type: Directory
    label: "SPRM output"

steps:
  ome_tiff_normalize_expr:
    in:
      data_dir:
        source: image_dir
    out: [output_dir]
    run: ome-tiff-normalize/ome_tiff_normalize.cwl
  ome_tiff_normalize_mask:
    in:
      data_dir:
        source: mask_dir
    out: [output_dir]
    run: ome-tiff-normalize/ome_tiff_normalize.cwl
  sprm:
    in:
      image_dir:
        source: ome_tiff_normalize_expr/output_dir
      mask_dir:
        source: ome_tiff_normalize_mask/output_dir
      options_file:
        source: options_file
      cell_types_directory:
        source: cell_types_directory
      options_preset:
        source: options_preset
      enable_manhole:
        source: enable_manhole
      enable_faulthandler:
        source: enable_faulthandler
      verbose:
        source: verbose
      processes:
        source: processes
    out: [output_dir]
    run: steps/sprm.cwl
    label: "SPRM analysis"
