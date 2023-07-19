import faulthandler
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from subprocess import CalledProcessError, check_output

from .outlinePCA import (
    bin_pca,
    getcellshapefeatures,
    getparametricoutline,
    kmeans_cluster_shape,
    pca_recon,
)
from .single_method_eval import *
from .single_method_eval_3D import *
from .SPRM_pkg import *

"""

Function:  Spatial Pattern and Relationship Modeling for HubMap common imaging pipeline
Inputs:    channel OME-TIFFs in "img_hubmap" folder
           paired segmentation OME-TIFFs in "mask_hubmap" folder
Returns:   OME-TIFF, CSV and PNG Files
Purpose:   Calculate various features and clusterings for multichannel images
Authors:   Ted (Ce) Zhang and Robert F. Murphy
Version:   1.03
01/21/2020 - 06/25/2020


"""

DEFAULT_OUTPUT_PATH = Path("sprm_outputs")
DEFAULT_OPTIONS_FILE = Path(__file__).parent / "options.txt"
DEFAULT_TEMP_DIRECTORY = Path("temp")
DOCKER_GIT_VERSION_PATH = Path("/opt/sprm-git-revision.json")


def get_sprm_version() -> str:
    """
    Returns the SPRM version. Tries three possibilities to find this
    information, in order:

    1) From a JSON file written into the Docker image
    2) Running 'git describe'
    3) reading from version.txt

    If all of these fail, returns 'unknown'.
    """
    if DOCKER_GIT_VERSION_PATH.is_file():
        with open(DOCKER_GIT_VERSION_PATH) as f:
            version_data = json.load(f)
            return version_data["version"]

    # maybe run from the repo directory
    directory = Path(__file__).parent
    try:
        return (
            check_output(
                ["git", "describe", "--dirty", "--always", "--abbrev=12"],
                cwd=directory,
            )
            .decode()
            .strip()
        )
    except CalledProcessError:
        pass

    try:
        with importlib.resources.open_text("sprm", "version.txt") as f:
            return f.read().strip()
    except Exception:
        pass

    return "unknown"


def analysis(
    img_file: Path,
    mask_file: Path,
    optional_img_file: Optional[Path],
    output_dir: Path,
    options: Dict[str, Any],
    celltype_labels: Path,
) -> Optional[Tuple[IMGstruct, MaskStruct, int, Optional[Dict[str, Any]]]]:
    image_stime = time.monotonic()
    print("Reading in image and corresponding mask file...")
    print("Image name:", img_file.name)

    im = IMGstruct(img_file, options)
    if options.get("debug"):
        print("Image dimensions: ", im.get_data().shape)

    # init cell features
    covar_matrix = []
    mean_vector = []
    total_vector = []

    # hot fix for stitched images pipeline
    # if there are scenes or time points - they should be channels
    # if im.get_data().shape[0] > 1 and len(im.get_channel_labels()) > 1:
    #     # data = im.get_data()[0, 0, :, :, :, :]
    #     # data = data[np.newaxis, np.newaxis, :, :, :, :]
    #     data = im.get_data()
    #     s, t, c, z, y, x = data.shape
    #     data = data.reshape(c, t, s, z, y, x)
    #     im.set_data(data)

    # get base file name for all output files
    baseoutputfilename = im.get_name()

    mask = MaskStruct(mask_file, options)

    # hot fix for stitched images pipeline
    # if there are scenes or time points - they should be channels
    # if mask.get_data().shape[0] > 1 and len(mask.get_channel_labels()) > 1:
    #     # data = im.get_data()[0, 0, :, :, :, :]
    #     # data = data[np.newaxis, np.newaxis, :, :, :, :]
    #     data = mask.get_data()
    #     s, t, c, z, y, x = data.shape
    #     data = data.reshape(c, t, s, z, y, x)
    #     mask.set_data(data)

    # switch channels and z dims
    ##############################
    ##############################
    ###LOCAL TESTING ON MAC###
    # data = im.get_data()
    # s, t, c, z, y, x = data.shape
    # data = data.reshape(s, t, z, c, y, x)
    # im.set_data(data)
    #
    # data = mask.get_data()
    # s, t, c, z, y, x = data.shape
    # data = data.reshape(s, t, z, c, y, x)
    # mask.set_data(data)
    # mask.set_bestz([0])
    ##############################
    ##############################

    # 0 == just sprm, 1 == segeval, 2 == both
    eval_pathway = options.get("sprm_segeval_both")

    seg_metrics = None
    if eval_pathway:
        if options.get("image_dimension") == "3D":
            seg_metrics = single_method_eval_3D(im, mask, output_dir)
        else:
            seg_metrics = single_method_eval(im, mask, output_dir)
        if eval_pathway == 1:
            # only perform evaluation on single segmentation method
            struct = {"Segmentation Evaluation Metrics v1.5": seg_metrics}

            with open(
                output_dir / (im.name + "-SPRM_Image_Quality_Measures.json"), "w"
            ) as json_file:
                json.dump(struct, json_file, indent=4, sort_keys=True, cls=NumpyEncoder)
            print("Finished Segmentation Evaluation for", im.path)
            # loop to next image
            return

    # combination of mask_img & get_masked_imgs
    ROI_coords = get_coordinates(mask, options)
    mask.set_ROI(ROI_coords)

    # quality control of image and mask for edge cells and best z slices +- n options
    quality_control(mask, im, ROI_coords, options)

    # get cells to be processed
    inCells = mask.get_interior_cells()
    cellidx = mask.get_cell_index()
    cell_count = len(inCells)

    # save cell graphs
    cell_graphs(mask, ROI_coords, inCells, baseoutputfilename, output_dir, options)

    # signal to noise ratio of the image
    SNR(im, baseoutputfilename, output_dir, cellidx, options)

    bestz = mask.get_bestz()
    # empty mask skip tile
    if not bestz and options.get("skip_empty_mask") == 1:
        print("Skipping tile...(mask is empty)")
        return
    # if len(bestz) > 1:
    #     zslices = mask.get_bestz()

    # check for whether there are accompanying images of the same field
    # TODO: don't require empty list if no other file
    opt_img_file = optional_img_file or []

    # NMF calculation
    NMF_calc(im, baseoutputfilename, output_dir, options)

    # do superpixel and PCA analysis before reallocating images to conserve memory
    # these are done on the whole image, not the individual cells
    # do clustering on the individual pixels to find 'pixel types'
    superpixels = voxel_cluster(im, options)
    plot_img(superpixels, bestz, baseoutputfilename + "-Superpixels.png", output_dir, options)

    # do PCA on the channel values to find channel components
    reducedim = clusterchannels(im, baseoutputfilename, output_dir, inCells, options)
    PCA_img = plotprincomp(
        reducedim, bestz, baseoutputfilename + "-Top3ChannelPCA.png", output_dir, options
    )

    # writing out as a ometiff file of visualizations by channels
    write_ometiff(im, output_dir, options, PCA_img, superpixels[bestz])

    # check if the image and mask spatial resolutions match
    # and reallocate intensity to the mask resolution if not
    # also merge in optional additional image if present
    reallocate_and_merge_intensities(im, mask, opt_img_file, options)
    # generate_fake_stackimg(im, mask, opt_img_file, options)

    if options.get("skip_texture"):
        # make fake textures matrix - all zeros
        textures = [
            np.zeros((1, 2, cell_count, len(im.channel_labels) * 6, 1)),
            im.channel_labels * 12,
        ]
        # save it
        for i in range(2):
            df = pd.DataFrame(
                textures[0][0, i, :, :, 0],
                columns=textures[1][: len(im.channel_labels) * 6],
                index=list(range(1, len(inCells) + 1)),
            )
            df.index.name = "ID"
            df.to_csv(
                output_dir / (baseoutputfilename + "-" + mask.channel_labels[i] + "_1_texture.csv")
            )
    else:
        textures = glcmProcedure(im, mask, output_dir, baseoutputfilename, ROI_coords, options)

    # time point loop (don't expect multiple time points)
    for t in range(0, im.get_data().shape[1]):
        seg_n = mask.get_labels("cell")
        # debug of cell_coordinates
        # if options.get("debug"): cell_coord_debug(mask, seg_n, options.get("num_outlinepoints"))

        # get normalized shape representation of each cell
        if not options.get("skip_outlinePCA"):
            outline_vectors, cell_polygons = getparametricoutline(mask, seg_n, ROI_coords, options)
            shape_vectors, norm_shape_vectors, pca = getcellshapefeatures(outline_vectors, options)
            if options.get("debug"):
                # just for testing
                kmeans_cluster_shape(shape_vectors, outline_vectors, output_dir, options)
                bin_pca(norm_shape_vectors, 1, outline_vectors, baseoutputfilename, output_dir)
                pca_recon(norm_shape_vectors, 1, pca, baseoutputfilename, output_dir)
                # pca_cluster_shape(shape_vectors, cell_polygons, output_dir, options)  # just for testing
            write_cell_polygs(
                cell_polygons,
                outline_vectors,
                cellidx,
                baseoutputfilename,
                output_dir,
                options,
            )
        else:
            print("Skipping outlinePCA...")
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
                (
                    covar_matrix[t, j, i, :, :],
                    mean_vector[t, j, i, :, :],
                    total_vector[t, j, i, :, :],
                ) = calculations(masked_imgs_coord[i], im, t, i, bestz)

        if not options.get("skip_outlinePCA"):
            # save the means, covars, shape and total for each cell
            save_all(
                baseoutputfilename,
                im,
                mask,
                output_dir,
                cellidx,
                options,
                mean_vector,
                covar_matrix,
                total_vector,
                shape_vectors,
                norm_shape_vectors,
            )

            # do cell analyze
            cell_analysis(
                im,
                mask,
                options,
                baseoutputfilename,
                bestz,
                output_dir,
                seg_n,
                cellidx,
                options,
                celltype_labels,
                mean_vector,
                covar_matrix,
                total_vector,
                shape_vectors,
                textures,
                norm_shape_vectors,
            )
        else:
            # same functions as above just without shape outlines
            save_all(
                baseoutputfilename,
                im,
                mask,
                output_dir,
                cellidx,
                options,
                mean_vector,
                covar_matrix,
                total_vector,
            )
            cell_analysis(
                im,
                mask,
                baseoutputfilename,
                bestz,
                output_dir,
                seg_n,
                cellidx,
                options,
                mean_vector,
                covar_matrix,
                total_vector,
                textures,
            )

    # mask.quit()
    # im.quit()

    if options.get("debug"):
        print(f"Runtime for image {im.name}: {time.monotonic() - image_stime}")

    return im, mask, cell_count, seg_metrics


def main(
    img_dir: Path,
    mask_dir: Path,
    processes: int,
    output_dir: Path = DEFAULT_OUTPUT_PATH,
    options_path: Path = DEFAULT_OPTIONS_FILE,
    optional_img_dir: Optional[Path] = None,
    celltype_labels: Path = None,
):
    sprm_version = get_sprm_version()
    print("SPRM", sprm_version)

    # read in options.txt
    options = read_options(options_path, DEFAULT_OPTIONS_FILE)

    # store results in a dir
    check_output_dir(output_dir, options)

    with open(output_dir / "sprm_version.txt", "w") as f:
        print(sprm_version, file=f)

    # get_imgs sorts to ensure the order of images and masks matches
    img_files = get_paths(img_dir)
    mask_files = get_paths(mask_dir)

    if optional_img_dir:
        opt_img_files = get_paths(optional_img_dir)
    else:
        opt_img_files = [None] * len(img_files)

    # init list of saved
    im_list = []
    mask_list = []
    cell_total = []
    seg_metric_list = []

    # start time of processing a single img
    stime = time.monotonic() if options.get("debug") else None

    ### LOCAL TESTING ###
    for i in range(len(img_files)):
        im, mask, cc, segm = analysis(
            img_files[i],
            mask_files[i],
            opt_img_files[0],
            output_dir,
            options,
            celltype_labels,
        )

        im_list.append(im)
        mask_list.append(mask)
        cell_total.append(cc)
        seg_metric_list.append(segm)
    #### END LOCAL TESTING ###

    ### CWL RUNS ###
    # use_subprocess_isolation = len(img_files) > 1 and not options.get("debug")
    # executor = ProcessPoolExecutor if use_subprocess_isolation else ThreadPoolExecutor
    # print("Using", processes, "worker(s) with executor", executor.__name__)
    # with executor(max_workers=processes) as executor:
    #     futures = []
    #     for img_file, mask_file, opt_img_file in zip(img_files, mask_files, opt_img_files):
    #         futures.append(
    #             executor.submit(
    #                 analysis,
    #                 img_file,
    #                 mask_file,
    #                 opt_img_file,
    #                 output_dir,
    #                 options,
    #                 celltype_labels,
    #             )
    #         )
    #
    #     for future in futures:
    #         maybe_result = future.result()
    #         if maybe_result is not None:
    #             im, mask, cell_count, seg_metrics = maybe_result
    #
    #             im_list.append(im)
    #             mask_list.append(mask)
    #             cell_total.append(cell_count)
    #             seg_metric_list.append(seg_metrics)

    ### CWL END ###

    if options.get("image_dimension") == "3D":
        quality_measures_3D(
            im_list, mask_list, seg_metric_list, cell_total, img_files, output_dir, options
        )
    else:
        quality_measures(
            im_list, mask_list, seg_metric_list, cell_total, img_files, output_dir, options
        )

    if options.get("debug"):
        print(f"Total runtime: {time.monotonic() - stime}")

    # recluster features
    # recluster(output_dir, im, options)


def argparse_wrapper():
    p = ArgumentParser()
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--optional-img-dir", type=Path, nargs="?")
    p.add_argument("-p", "--processes", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--enable-manhole", action="store_true")
    p.add_argument("--enable-faulthandler", action="store_true")
    p.add_argument("--celltype-labels", type=Path, default=None)

    options_file_group = p.add_mutually_exclusive_group()
    options_file_group.add_argument("--options-file", type=Path, default=DEFAULT_OPTIONS_FILE)
    options_file_group.add_argument("--options-preset")

    argss = p.parse_args()

    if argss.enable_manhole:
        import manhole

        manhole.install(activate_on="USR1")

    if argss.enable_faulthandler:
        faulthandler.enable(all_threads=True)

    if argss.options_preset is not None:
        argss.options_file = DEFAULT_OPTIONS_FILE.with_name(f"options-{argss.options_preset}.txt")

    main(
        img_dir=argss.img_dir,
        mask_dir=argss.mask_dir,
        processes=argss.processes,
        output_dir=argss.output_dir,
        options_path=argss.options_file,
        optional_img_dir=argss.optional_img_dir,
        celltype_labels=argss.celltype_labels,
    )
