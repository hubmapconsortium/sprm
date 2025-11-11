import faulthandler
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from subprocess import CalledProcessError, check_output

from .outlinePCA import (
    bin_pca,
    get_parametric_outline,
    getcellshapefeatures,
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
    celltype_labels: Optional[pd.DataFrame],
) -> Optional[Tuple[IMGstruct, MaskStruct, int, Optional[Dict[str, Any]]]]:
    """
    Legacy analysis function - now uses modular API internally.
    
    This function maintains backward compatibility with the original SPRM interface
    while leveraging the new modular architecture underneath.
    """
    image_stime = time.monotonic()
    
    # Import modular API
    from .SPRM_pkg import compute_cell_centers
    from .modules import preprocessing, segmentation_eval, shape_analysis
    from .modules import spatial_graphs, image_analysis, cell_features, clustering
    from .modules.checkpoint_manager import CheckpointManager, CoreData, ShapeData
    
    # Module 1: Preprocessing
    # Don't use checkpoints here to maintain legacy behavior
    print("Reading in image and corresponding mask file...")
    print("Image name:", img_file.name)
    
    im = IMGstruct(img_file, options)
    if options.get("debug"):
        print("Image dimensions: ", im.get_data().shape)
    
    mask = MaskStruct(mask_file, options)
    
    # Extract ROI coords and QC
    ROI_coords = get_coordinates(mask, options)
    mask.set_ROI(ROI_coords)
    quality_control(mask, im, ROI_coords, options)
    
    # Build CoreData manually (don't save checkpoint for legacy mode)
    cell_centers = compute_cell_centers(ROI_coords)
    core_data = CoreData(
        im=im,
        mask=mask,
        roi_coords=ROI_coords,
        interior_cells=mask.get_interior_cells(),
        edge_cells=mask.get_edge_cells(),
        cell_index=mask.get_cell_index(),
        bad_cells=mask.get_bad_cells(),
        bestz=mask.get_bestz(),
        cell_centers=cell_centers,
    )
    
    # Module 2: Segmentation Evaluation (if requested)
    eval_pathway = options.get("sprm_segeval_both")
    seg_metrics = None
    
    if eval_pathway:
        if options.get("image_dimension") == "3D":
            seg_metrics = single_method_eval_3D(im, mask, output_dir)
        else:
            seg_metrics = single_method_eval(im, mask, output_dir)
            
        if eval_pathway == 1:
            # Only perform evaluation, then exit
            struct = {"Segmentation Evaluation Metrics v1.5": seg_metrics}
            with open(
                output_dir / (im.name + "-SPRM_Image_Quality_Measures.json"), "w"
            ) as json_file:
                json.dump(struct, json_file, indent=4, sort_keys=True, cls=NumpyEncoder)
            print("Finished Segmentation Evaluation for", im.path)
            return
    
    # Check for empty mask
    bestz = mask.get_bestz()
    if not bestz and options.get("skip_empty_mask") == 1:
        print("Skipping tile...(mask is empty)")
        return
    
    # Module 3: Shape Analysis (if requested)
    shape_data = None
    shape_vectors = None
    norm_shape_vectors = None
    
    if options.get("run_outlinePCA"):
        seg_n = mask.get_labels("cell")
        cellidx = mask.get_cell_index()
        baseoutputfilename = im.get_name()
        
        outline_vectors, cell_polygons = get_parametric_outline(
            mask, seg_n, ROI_coords, options
        )
        shape_vectors, norm_shape_vectors, pca = getcellshapefeatures(outline_vectors, options)
        
        if options.get("debug"):
            kmeans_cluster_shape(shape_vectors, outline_vectors, output_dir, options)
            bin_pca(norm_shape_vectors, 1, outline_vectors, baseoutputfilename, output_dir)
            pca_recon(norm_shape_vectors, 1, pca, baseoutputfilename, output_dir)
        
        write_cell_polygs(
            cell_polygons, outline_vectors, cellidx, 
            baseoutputfilename, output_dir, options
        )
        
        shape_data = ShapeData(
            shape_vectors=shape_vectors,
            norm_shape_vectors=norm_shape_vectors,
            outline_vectors=outline_vectors,
            cell_polygons=cell_polygons,
        )
    else:
        print("Skipping outlinePCA...")
    
    # Module 4a: Spatial Graphs (if requested)
    if options.get("cell_graph"):
        inCells = mask.get_interior_cells()
        cellidx = mask.get_cell_index()
        baseoutputfilename = im.get_name()
        
        cell_graphs(mask, ROI_coords, inCells, baseoutputfilename, output_dir, options)
        SNR(im, baseoutputfilename, output_dir, cellidx, options)
    
    # Module 4b: Image Analysis (if requested)
    if options.get("image_analysis"):
        baseoutputfilename = im.get_name()
        inCells = mask.get_interior_cells()
        
        NMF_calc(im, baseoutputfilename, output_dir, options)
        superpixels = voxel_cluster(im, options)
        plot_img(superpixels, bestz, baseoutputfilename + "-Superpixels.png", output_dir, options)
        
        reducedim = clusterchannels(im, baseoutputfilename, output_dir, inCells, options)
        PCA_img = plotprincomp(
            reducedim, bestz, baseoutputfilename + "-Top3ChannelPCA.png", output_dir, options
        )
        write_ometiff(im, output_dir, options, PCA_img, superpixels[bestz])
    
    # Module 5: Cell Features
    # Reallocate intensities
    opt_img_file = optional_img_file or []
    reallocate_and_merge_intensities(im, mask, opt_img_file, options)
    
    # Compute texture or create zeros
    inCells = mask.get_interior_cells()
    cell_count = len(inCells)
    
    if options.get("skip_texture"):
        textures = [
            np.zeros((1, 2, cell_count, len(im.channel_labels) * 6, 1)),
            im.channel_labels * 12,
        ]
        baseoutputfilename = im.get_name()
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
        baseoutputfilename = im.get_name()
        textures = glcmProcedure(im, mask, output_dir, baseoutputfilename, ROI_coords, options)
    
    # Compute cell statistics
    covar_matrix = []
    mean_vector = []
    total_vector = []
    
    for t in range(0, im.get_data().shape[1]):
        for j in range(0, mask.get_data().shape[2]):
            masked_imgs_coord = ROI_coords[j]
            masked_imgs_coord = [masked_imgs_coord[i] for i in inCells]
            
            covar_matrix = build_matrix(im, mask, masked_imgs_coord, j, covar_matrix)
            mean_vector = build_vector(im, mask, masked_imgs_coord, j, mean_vector)
            total_vector = build_vector(im, mask, masked_imgs_coord, j, total_vector)
            
            for i in range(0, len(masked_imgs_coord)):
                (
                    covar_matrix[t, j, i, :, :],
                    mean_vector[t, j, i, :, :],
                    total_vector[t, j, i, :, :],
                ) = calculations(masked_imgs_coord[i], im, t, i, bestz)
    
    # Save all features
    cellidx = mask.get_cell_index()
    baseoutputfilename = im.get_name()
    save_all(
        filename=baseoutputfilename,
        im=im,
        mask=mask,
        output_dir=output_dir,
        cellidx=cellidx,
        options=options,
        mean_vector=mean_vector,
        covar_matrix=covar_matrix,
        total_vector=total_vector,
        outline_vectors=shape_vectors,
        norm_shape_vectors=norm_shape_vectors,
    )
    
    # Module 6: Clustering
    seg_n = mask.get_labels("cell")
    df_all_cluster_list = []
    
    cell_analysis(
        im=im,
        mask=mask,
        filename=baseoutputfilename,
        bestz=bestz,
        output_dir=output_dir,
        seg_n=seg_n,
        cellidx=cellidx,
        options=options,
        celltype_labels=celltype_labels,
        df_all_cluster_list=df_all_cluster_list,
        mean_vector=mean_vector,
        covar_matrix=covar_matrix,
        total_vector=total_vector,
        texture_vectors=textures[0],
        texture_channels=textures[1],
        shape_vectors=shape_vectors,
        norm_shape_vectors=norm_shape_vectors,
    )
    
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
    celltype_labels: Optional[list[Path]] = None,
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

    # subtype
    if celltype_labels is None:
        cell_types_by_image = [None] * len(img_files)
    else:
        cell_types_by_image = collect_parse_cell_types(celltype_labels)

    # init list of saved
    im_list = []
    mask_list = []
    cell_total = []
    seg_metric_list = []

    # start time of processing a single img
    stime = time.monotonic() if options.get("debug") else None

    ### LOCAL TESTING ###
    # for i in range(len(img_files)):
    #     im, mask, cc, segm = analysis(
    #         img_files[i],
    #         mask_files[i],
    #         opt_img_files[0],
    #         output_dir,
    #         options,
    #         cell_types_by_image[i],
    #     )
    #
    #     im_list.append(im)
    #     mask_list.append(mask)
    #     cell_total.append(cc)
    #     seg_metric_list.append(segm)
    #### END LOCAL TESTING ###

    ### CWL RUNS ###
    use_subprocess_isolation = len(img_files) > 1 and not options.get("debug")
    executor = ProcessPoolExecutor if use_subprocess_isolation else ThreadPoolExecutor
    print("Using", processes, "worker(s) with executor", executor.__name__)
    with executor(max_workers=processes) as executor:
        futures = []
        for img_file, mask_file, opt_img_file, cell_types in zip(
            img_files, mask_files, opt_img_files, cell_types_by_image
        ):
            futures.append(
                executor.submit(
                    analysis,
                    img_file,
                    mask_file,
                    opt_img_file,
                    output_dir,
                    options,
                    cell_types,
                )
            )

        for future in futures:
            maybe_result = future.result()
            if maybe_result is not None:
                im, mask, cell_count, seg_metrics = maybe_result

                im_list.append(im)
                mask_list.append(mask)
                cell_total.append(cell_count)
                seg_metric_list.append(seg_metrics)

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
    p.add_argument(
        "--threadpool-limit",
        type=int,
        help="Limit the number of threads used by BLAS/OpenMP libraries (e.g., MKL)",
    )
    p.add_argument("--celltype-labels", type=Path, action="append")

    options_file_group = p.add_mutually_exclusive_group()
    options_file_group.add_argument("--options-file", type=Path, default=DEFAULT_OPTIONS_FILE)
    options_file_group.add_argument("--options-preset")

    argss = p.parse_args()

    if argss.enable_manhole:
        import manhole

        manhole.install(activate_on="USR1")

    if argss.enable_faulthandler:
        faulthandler.enable(all_threads=True)

    if argss.threadpool_limit is not None:
        import threadpoolctl

        threadpoolctl.threadpool_limits(limits=argss.threadpool_limit)
        print(f"Limited threadpool to {argss.threadpool_limit} threads")

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
