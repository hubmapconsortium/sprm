cwlVersion: v1.1
class: CommandLineTool
label: SPRM analysis
hints:
  DockerRequirement:
    dockerPull: hubmap/sprm:latest
  NetworkAccess:
    networkAccess: true
baseCommand:

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
  num_dims:
    type: int?
    default: 2
    inputBinding:
      position: 3
      prefix: "--num-dims"

outputs:
  sdata_zarr:
    type: Directory
    outputBinding:
      glob: sprm_output.zarr
