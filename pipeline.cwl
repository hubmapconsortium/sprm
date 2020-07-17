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

outputs:
  sprm_output:
    outputSource: sprm/output_dir
    type: Directory
    label: "SPRM output"

steps:
  sprm:
    in:
      image_dir:
        source: image_dir
      mask_dir:
        source: mask_dir
      options_file:
        source: options_file
    out: [output_dir]
    run: steps/sprm.cwl
    label: "SPRM analysis"
