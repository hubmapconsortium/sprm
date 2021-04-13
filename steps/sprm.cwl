cwlVersion: v1.1
class: CommandLineTool
label: SPRM analysis
hints:
  DockerRequirement:
    dockerPull: hubmap/sprm:1.0.2-post2
  NetworkAccess:
    networkAccess: true
baseCommand: sprm

inputs:
  image_dir:
    type: Directory
    inputBinding:
      position: 0
  mask_dir:
    type: Directory
    inputBinding:
      position: 1
  enable_manhole:
    type: boolean?
    inputBinding:
      position: 2
      prefix: "--enable-manhole"
  options_file:
    type: File?
    inputBinding:
      position: 3
      prefix: "--options-file"
outputs:
  output_dir:
    type: Directory
    outputBinding:
      glob: sprm_outputs
