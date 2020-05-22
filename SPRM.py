from SPRM_pkg import *
from outlinePCA import getparametricoutline, getcellshapefeatures
from argparse import ArgumentParser

"""

Function:  Spatial Pattern and Relationship Modeling for HubMap common imaging pipeline
Inputs:    channel OME-TIFFs in "img_hubmap" folder
           paired segmentation OME-TIFFs in "mask_hubmap" folder
Returns:   OME-TIFF, CSV and PNG Files
Purpose:   Calculate various features and clusterings for multichannel images
Authors:   Ted Zhang and Robert F. Murphy
Version:   0.55
01/21/2020 - 05/22/2020
 

"""


def main(img_dir: Path, mask_dir: Path, output_dir: Path, options_path: Path):
    # get_imgs sorts to ensure the order of images and masks matches
    img_files = get_paths(img_dir)
    mask_files = get_paths(mask_dir)

    # read in options.txt
    options = read_options(options_path)

    covar_matrix = []
    mean_vector = []
    total_vector = []
    cell_total = []

    # store results in a dir
    check_output_dir(output_dir, options)
    # loop of img files
    for idx in range(0, len(img_files)):

        print('Reading in image and corresponding mask files...')
        img_file = img_files[idx]
        print('Image name: ', img_file.name)

        im = IMGstruct(img_file, options)
        if options.get("debug"): print('Image dimensions: ', im.get_data().shape)

        mask_file = mask_files[idx]
        mask = MaskStruct(mask_file, options)

        bestz = mask.get_bestz()

        cell_total.append(np.amax(mask.get_data()))

        # start time of processing a single img
        stime = time.monotonic() if options.get("debug") else None

        # time point loop (don't expect multiple time points)
        for t in range(0, im.get_data().shape[1]):
            # if bestz is None or np.max(mask.get_data()) < 2: 
            if bestz is None and options.get('skip_empty_mask') is 1:
                print('Skipping tile...(mask is empty)')
                break

            if options.get("debug"): print('IN TIMEPOINTS LOOP ' + str(t))
            # get base file name for all output files
            baseoutputfilename = im.get_name()
            if options.get("debug"): print('filename: ', baseoutputfilename)

            # do clustering on the individual pixels to find 'pixel types'
            superpixels = voxel_cluster(im, options)
            plot_img(superpixels[bestz], baseoutputfilename + '-Superpixels.png', output_dir)

            # do PCA on the channel values to find channel components
            reducedim = clusterchannels(im, options)
            PCA_img = plotprincomp(reducedim, bestz, baseoutputfilename + '-Top3ChannelPCA.png', output_dir, options)

            # writing out as a ometiff file of visualizations by channels
            write_ometiff(im, output_dir, PCA_img, superpixels[bestz])

            seg_n = mask.get_labels('cells')
            # debug of cell_coordinates
            # if options.get("debug"): cell_coord_debug(mask, seg_n, options.get("num_outlinepoints"))

            # get normalized shape representation of each cell
            outline_vectors, cell_polygons = getparametricoutline(mask, seg_n, options)
            shape_vectors = getcellshapefeatures(outline_vectors, options)
            write_cell_polygs(cell_polygons, baseoutputfilename, output_dir, options)

            # signal to noise ratio of the image
            SNR(im, baseoutputfilename, output_dir, options)
            # loop of types of segmentation (channels in the mask img)
            for j in range(0, mask.get_data().shape[2]):
                # get the mask for this particular segmentation
                # (e.g., cells, nuclei...)
                labeled_mask, maskIDs = mask_img(mask, j)
                # convert indexed image into lists of pixels
                # in each object (ROI) of this segmentation
                masked_imgs_coord = get_masked_imgs(labeled_mask, maskIDs)
                # make the matrix and vectors to hold calculations
                covar_matrix = build_matrix(im, mask, masked_imgs_coord, j, covar_matrix)
                mean_vector = build_vector(im, mask, masked_imgs_coord, j, mean_vector)
                total_vector = build_vector(im, mask, masked_imgs_coord, j, total_vector)
        
                # loop of ROIs
                for i in range(0, len(masked_imgs_coord)):
                    covar_matrix[t, j, i, :, :], mean_vector[t, j, i, :, :], total_vector[t, j, i, :, :] = calculations(
                        masked_imgs_coord[i], im, t, i)
        
            # save the means, covars, shape and total for each cell
            save_all(baseoutputfilename, im, seg_n, output_dir, options, mean_vector, covar_matrix, total_vector,
                      shape_vectors)
        
            # do cell analyze
            cell_analysis(im, mask, baseoutputfilename, bestz, seg_n, output_dir, options, mean_vector, covar_matrix,
                          total_vector,
                          shape_vectors)
        
        if options.get("debug"): print('Per image runtime: ' + str(time.monotonic() - stime))
        print('Finished analyzing ' + str(idx + 1) + ' image(s)')
        mask.quit()
        im.quit()

    # summary of all tiles/files in a single run
    summary(im, cell_total, img_files, output_dir, options)



if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('img_dir', type=Path)
    p.add_argument('mask_dir', type=Path)
    p.add_argument('output_dir', type=Path)
    p.add_argument('options_file', type=Path)

    argss = p.parse_args()

    main(argss.img_dir, argss.mask_dir, argss.output_dir, argss.options_file)
