from SPRM_pkg import *
import sys

"""

Function:  Spatial Pattern and Relationship Modeling for HubMap common imaging pipeline
Inputs:    channel OME-TIFFs in "img_hubmap" folder
           paired segmentation OME-TIFFs in "mask_hubmap" folder
Returns:   OME-TIFF, CSV and PNG Files
Purpose:   Calculate various features and clusterings for multichannel images
Authors:   Ted Zhang and Robert F. Murphy
Version:   0.53
01/21/2020 - 04/21/2020
 

"""


def main(img_dir,mask_dir,options_path):
    # get_imgs sorts to ensure the order of images and masks matches
    img_files = get_imgs(img_dir)
    mask_files = get_imgs(mask_dir)
    
    #read in options.txt 
    options = read_options(options_path)
    
    covar_matrix = []
    mean_vector = []
    total_vector = []
        
    #store results in a dir 
    cwd = check_results_dir()
    
    #loop of img files
    for idx in range(0, len(img_files)):
        print('IN IMAGE FILE LOOP')
        img_file = img_files[idx]
        print(img_file)
        im = IMGstruct(img_file, options)
        print(im.get_data().shape)

        mask_file = mask_files[idx]
        mask = MaskStruct(mask_file, options)

        bestz = mask.get_bestz()
        
        #start time of processing a single img
        stime = time.time()
        
        # time point loop (don't expect multiple time points)
        for t in range(0, im.get_data().shape[1]):
            print('IN TIMEPOINTS LOOP')
            # get base file name for all output files
            baseoutputfilename = im.get_name()
            print(baseoutputfilename)
   
            # do clustering on the individual pixels to find 'pixel types'
            superpixels = voxel_cluster(im, options)
            plot_img(superpixels[bestz], baseoutputfilename + '-Superpixels.png')

            #do PCA on the channel values to find channel components
            reducedim = clusterchannels(im, options)
            PCA_img = plotprincomp(reducedim, bestz, baseoutputfilename + '-Top3ChannelPCA.png')

            # writing out as a ometiff file of visualizations by channels
            print('Writing out ometiffs for visualizations...')
            write_ometiff(im,PCA_img, superpixels[bestz])
   
            seg_n = mask.get_labels('cells')
            # get normalized shape representation of each cell
            outline_vectors = getparametricoutline(mask, seg_n, options.get("num_outlinepoints"))
            shape_vectors = getcellshapefeatures(outline_vectors, options)

            # loop of types of segmentation (channels in the mask img)
            for j in range(0,mask.get_data().shape[2]):
                print('IN SEGMENTATION LOOP')
                # get the mask for this particular segmentation
                # (e.g., cells, nuclei...)
                labeled_mask, maskIDs = mask_img(mask, mask_dir, j, options)
                # convert indexed image into lists of pixels
                # in each object (ROI) of this segmentation
                masked_imgs_coord = get_masked_imgs(labeled_mask, maskIDs)
                # make the matrix and vectors to hold calculations
                covar_matrix = build_matrix(im, mask, masked_imgs_coord, j, covar_matrix)
                mean_vector = build_vector(im, mask, masked_imgs_coord, j, mean_vector)
                total_vector = build_vector(im, mask, masked_imgs_coord, j, total_vector)

                # loop of ROIs
                for i in range(0, len(masked_imgs_coord)):
                    covar_matrix[t, j, i, :, :], mean_vector[t, j, i, :, :], total_vector[t, j, i, :, :] = calculations(masked_imgs_coord[i], im, t)
            
            # save the means, covars, shape and total for each cell
            print('Writing to csv all matrices...')
            save_all(baseoutputfilename, im, seg_n, options, mean_vector, covar_matrix, total_vector, shape_vectors)

            # do cell analyze
            cell_analysis(im, mask, baseoutputfilename, bestz, seg_n, options, mean_vector, covar_matrix, total_vector, shape_vectors)
       
        print('Per image runtime: ' + str(time.time() - stime))
    
        mask.quit()
        im.quit()
    os.chdir(cwd)

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3])
