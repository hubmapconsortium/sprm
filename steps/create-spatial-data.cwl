cwlVersion: v1.1
class: CommandLineTool
label: SPRM analysis
hints:
  DockerRequirement:
    dockerPull: hubmap/sprm-spatialdata-conversion:2.2.4
  NetworkAccess:
    networkAccess: true
baseCommand: /opt/SPRM_output_convert.py

inputs:
  image_dir:
    type: Directory[]
    inputBinding:
      position: 0
      prefix: "--img-dir"
  mask_dir:
    type: Directory[]
    inputBinding:
      position: 1
      prefix: "--mask-dir"
  sprm_dir:
    type: Directory
    inputBinding:
      position: 2
      prefix: "--sprm-dir"
  spatialdata_dir:
    type: Directory
    inputBinding:
      position: 3
      prefix: "--spatialdata-dir"

outputs:
  sdata_zarrs:
    type: Directory[]
    outputBinding:
      glob: "*sprm_output.zarr"
  segmentation_metadata_json:
    type: File
    outputBinding:
      glob: "segmentation-metadata.json"
