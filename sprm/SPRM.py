from argparse import ArgumentParser
from typing import Optional

from .SPRM_pkg import *
from .outlinePCA import getparametricoutline, getcellshapefeatures, pca_cluster_shape,  pca_recon, bin_pca

"""

Function:  Spatial Pattern and Relationship Modeling for HubMap common imaging pipeline
Inputs:    channel OME-TIFFs in "img_hubmap" folder
           paired segmentation OME-TIFFs in "mask_hubmap" folder
Returns:   OME-TIFF, CSV and PNG Files
Purpose:   Calculate various features and clusterings for multichannel images
Authors:   Ted (Ce) Zhang and Robert F. Murphy
Version:   0.80
01/21/2020 - 02/17/2020
 

"""

DEFAULT_OUTPUT_PATH = Path('sprm_outputs')
DEFAULT_OPTIONS_FILE = Path(__file__).parent / 'options.txt'
DEFAULT_TEMP_DIRECTORY = Path('temp')


def main(
        img_dir: Path,
        mask_dir: Path,
        output_dir: Path = DEFAULT_OUTPUT_PATH,
        options_path: Path = DEFAULT_OPTIONS_FILE,
        optional_img_dir: Optional[Path] = None,
):
    # get_imgs sorts to ensure the order of images and masks matches
    img_files = get_paths(img_dir)
    mask_files = get_paths(mask_dir)

    if optional_img_dir:
        opt_img_files = get_paths(optional_img_dir)

    # read in options.txt
    options = read_options(options_path)

    # init cell features
    covar_matrix = []
    mean_vector = []
    total_vector = []
    cell_total = []

    # store results in a dir
    check_output_dir(output_dir, options)

    # start time of processing a single img
    stime = time.monotonic() if options.get("debug") else None

    # loop of img files
    for idx in range(0, len(img_files)):

        print('Reading in image and corresponding mask file...')
        img_file = img_files[idx]
        print('Image name: ', img_file.name)

        im = IMGstruct(img_file, options)
        if options.get("debug"): print('Image dimensions: ', im.get_data().shape)

        #hot fix for stitched images pipeline
        #if there are scenes or time points - they should be channels
        if im.get_data().shape[0] > 1 and len(im.get_channel_labels()) > 1:
            # data = im.get_data()[0, 0, :, :, :, :]
            # data = data[np.newaxis, np.newaxis, :, :, :, :]
            data = im.get_data()
            s, t, c, z, y, x = data.shape
            data = data.reshape(c, t, s, z, y, x)
            im.set_data(data)

        # get base file name for all output files
        baseoutputfilename = im.get_name()

        mask_file = mask_files[idx]
        mask = MaskStruct(mask_file, options)

        #hot fix for stitched images pipeline
        #if there are scenes or time points - they should be channels
        if mask.get_data().shape[0] > 1 and len(mask.get_channel_labels()) > 1:
            # data = im.get_data()[0, 0, :, :, :, :]
            # data = data[np.newaxis, np.newaxis, :, :, :, :]
            data = mask.get_data()
            s, t, c, z, y, x = data.shape
            data = data.reshape(c, t, s, z, y, x)
            mask.set_data(data)

        # combination of mask_img & get_masked_imgs
        ROI_coords = get_coordinates(mask, options)

        # quality control of image and mask for edge cells and best z slices +- n options
        quality_control(mask, im, ROI_coords, options)

        # get cells to be processed
        inCells = mask.get_interior_cells()
        cell_total.append(len(inCells))

        #save cell graphs
        cell_graphs(ROI_coords, inCells, baseoutputfilename, output_dir)

        # signal to noise ratio of the image
        SNR(im, baseoutputfilename, output_dir, options)

        bestz = mask.get_bestz()
        # empty mask skip tile
        if not bestz and options.get('skip_empty_mask') == 1:
            print('Skipping tile...(mask is empty)')
            continue
        # if len(bestz) > 1:
        #     zslices = mask.get_bestz()

        # check for whether there are accompanying images of the same field
        if optional_img_dir:
            opt_img_file = opt_img_files[idx]
        else:
            opt_img_file = []

        # do superpixel and PCA analysis before reallocating images to conserve memory
        # these are done on the whole image, not the individual cells
        # do clustering on the individual pixels to find 'pixel types'
        superpixels = voxel_cluster(im, options)
        plot_img(superpixels, bestz[0], baseoutputfilename + '-Superpixels.png', output_dir)

        # do PCA on the channel values to find channel components
        reducedim, pca_channels = clusterchannels(im, baseoutputfilename, output_dir, options)
        PCA_img = plotprincomp(reducedim, bestz[0], baseoutputfilename + '-Top3ChannelPCA.png', output_dir, options)

        # writing out as a ometiff file of visualizations by channels
        write_ometiff(im, output_dir, bestz, PCA_img, superpixels[bestz[0]])

        # check if the image and mask spatial resolutions match
        # and reallocate intensity to the mask resolution if not
        # also merge in optional additional image if present
        reallocate_and_merge_intensities(im, mask, opt_img_file, options)
        #generate_fake_stackimg(im, mask, opt_img_file, options)

        if options.get('skip_texture'):
            #make fake textures matrix - all zeros
            textures = [np.zeros((1, 2, cell_total[idx], len(im.channel_labels) * 6, 1)), im.channel_labels * 12]
            #save it
            for i in range(2):
                df = pd.DataFrame(textures[0][0, i, :, :, 0], columns=textures[1][:len(im.channel_labels) * 6], index=list(range(1, len(inCells) + 1)))
                df.index.name = 'ID'
                df.to_csv(output_dir / (baseoutputfilename + '-' + mask.channel_labels[i] + '_0_texture.csv'))
        else:
            textures = glcmProcedure(im, mask, bestz, output_dir, cell_total, baseoutputfilename, options)
        # generate_fake_stackimg(im, mask, opt_img_file, options)



        # time point loop (don't expect multiple time points)
        for t in range(0, im.get_data().shape[1]):

            seg_n = mask.get_labels('cells')
            # debug of cell_coordinates
            # if options.get("debug"): cell_coord_debug(mask, seg_n, options.get("num_outlinepoints"))

            # get normalized shape representation of each cell
            if not options.get('skip_outlinePCA'):
                outline_vectors, cell_polygons = getparametricoutline(mask, seg_n, ROI_coords, options)
                shape_vectors, pca = getcellshapefeatures(outline_vectors, options)
                if options.get('debug'):
                    bin_pca(shape_vectors, 1, cell_polygons, output_dir) #just for testing
                    pca_recon(shape_vectors, 1, pca, output_dir)  # just for testing
                    # pca_cluster_shape(shape_vectors, cell_polygons, output_dir, options)  # just for testing
                write_cell_polygs(cell_polygons, baseoutputfilename, output_dir, options)
            else:
                print('Skipping outlinePCA...')
            # loop of types of segmentation (channels in the mask img)
            for j in range(0, mask.get_data().shape[2]):
                # get the mask for this particular segmentation
                # (e.g., cells, nuclei...)
                # labeled_mask, maskIDs = mask_img(mask, j)
                # convert indexed image into lists of pixels
                # in each object (ROI) of this segmentation
                # masked_imgs_coord = get_masked_imgs(labeled_mask, maskIDs)
                # make the matrix and vectors to hold calculations

                masked_imgs_coord = ROI_coords[j]
                # get only the ROIs that are interior
                masked_imgs_coord = [masked_imgs_coord[i] for i in inCells]

                covar_matrix = build_matrix(im, mask, masked_imgs_coord, j, covar_matrix)
                mean_vector = build_vector(im, mask, masked_imgs_coord, j, mean_vector)
                total_vector = build_vector(im, mask, masked_imgs_coord, j, total_vector)

                # loop of ROIs
                for i in range(0, len(masked_imgs_coord)):
                    covar_matrix[t, j, i, :, :], mean_vector[t, j, i, :, :], total_vector[t, j, i, :, :] = calculations(
                        masked_imgs_coord[i], im, t, i)

            if not options.get('skip_outlinePCA'):
                # save the means, covars, shape and total for each cell
                save_all(baseoutputfilename, im, mask, output_dir, options, mean_vector, covar_matrix, total_vector,
                         shape_vectors)

                # do cell analyze
                cell_analysis(im, mask, baseoutputfilename, bestz, output_dir, seg_n, options, mean_vector,
                              covar_matrix,
                              total_vector,
                              shape_vectors, textures)
            else:
                # same functions as above just without shape outlines
                save_all(baseoutputfilename, im, seg_n, output_dir, options, mean_vector, covar_matrix, total_vector)
                cell_analysis(im, mask, baseoutputfilename, bestz, seg_n, output_dir, options, mean_vector,
                              covar_matrix, total_vector, textures)

        if options.get("debug"): print('Per image runtime: ' + str(time.monotonic() - stime))
        print('Finished analyzing ' + str(idx + 1) + ' image(s)')
        mask.quit()
        im.quit()

    # summary of all tiles/files in a single run
    summary(im, cell_total, img_files, output_dir, options)

    # recluster features
    # recluster(output_dir, im, options)


def argparse_wrapper():
    p = ArgumentParser()
    p.add_argument('img_dir', type=Path)
    p.add_argument('mask_dir', type=Path)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_PATH)
    p.add_argument('--options-file', type=Path, default=DEFAULT_OPTIONS_FILE)
    p.add_argument('optional_img_dir', type=Path, nargs='?', default=False)
    argss = p.parse_args()

    if argss.optional_img_dir:
        main(argss.img_dir, argss.mask_dir, argss.output_dir, argss.options_file, argss.optional_img_dir)
    else:
        main(argss.img_dir, argss.mask_dir, argss.output_dir, argss.options_file)
