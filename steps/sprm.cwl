cwlVersion: v1.1
class: CommandLineTool
label: SPRM analysis
hints:
  DockerRequirement:
    dockerPull: hubmap/sprm:1.0.9.3
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
  processes:
    type: int?
    default: 1
    inputBinding:
      position: 2
      prefix: "--processes"
  enable_manhole:
    type: boolean?
    inputBinding:
      position: 3
      prefix: "--enable-manhole"
  enable_faulthandler:
    type: boolean?
    inputBinding:
      position: 4
      prefix: "--enable-faulthandler"
  verbose:
    type: boolean?
    inputBinding:
      position: 5
      prefix: "--verbose"
  options_file:
    type: File?
    inputBinding:
      position: 128
      prefix: "--options-file"
  options_preset:
    type: string?
    inputBinding:
      position: 129
      prefix: "--options-preset"
outputs:
  output_dir:
    type: Directory
    outputBinding:
      glob: sprm_outputs
