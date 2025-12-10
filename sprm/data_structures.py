import logging
from pathlib import Path
from sys import getsizeof
from typing import Dict, Union, Any

import numpy as np
from aicsimageio import AICSImage

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

class IMGstruct:
    """
    Main Struct for IMG information
    """

    img: AICSImage
    data: np.ndarray
    path: Path
    name: str

    def __init__(self, path: Path, options):
        self.img = self.read_img(path, options)
        self.data = self.read_data(options)
        self.path = path
        self.name = path.name
        self.channel_labels = self.read_channel_names()
        self.channel_dict = {name: idx for idx, name in enumerate(self.channel_labels)}
        self.cached_data = {}
        
    def cache_set(self, key: str, val: Any)-> None:
        if isinstance(val, np.ndarray):
            LOGGER.info(f"size of cached object <{key}> is {getsizeof(val)} {val.nbytes}")
        else:
            LOGGER.info(f"size of cached object <{key}> type {type(val)} is {getsizeof(val)}")
        self.cached_data[key] = val

    def cache_get(self, key: str) -> Any:
        """
        returns None if the entry is missing
        """
        return self.cached_data.get(key)

    def get_img_channel_generator(self, z=None):
        if z is not None:
            if isinstance(z, int):
                for chan in range(self.img.dims.C):
                    rslt = np.expand_dims(self.get_plane(chan, z), axis=0)
                    LOGGER.debug(f"generator: {z} -> {chan} {z} {rslt.shape}")
                    yield chan, z, rslt
            elif isinstance(z, list):
                for chan in range(self.img.dims.C):
                    for z_idx in z:
                        rslt = np.expand_dims(self.get_plane(chan, z_idx), axis=0)
                        LOGGER.debug(f"generator: {z} -> {chan} {z_idx} {rslt.shape}")
                        yield chan, z_idx, rslt
            else:
                raise RuntimeError(f"z parameter is neither an int nor a list: {z}")
        else:
            for chan in range(self.img.dims.C):
                for z_idx in range(self.img.dims.Z):
                    rslt = np.expand_dims(self.get_plane(chan, z_idx), axis=0)
                    LOGGER.debug(f"generator: {z} -> {chan} {z_idx} {rslt.shape}")
                    yield chan, z_idx, rslt

    def apply_scale(self, channel: Union[int, str], factor: float) -> None:
        if factor == 1.0:
            return  # because the scaling has no effect
        if isinstance(channel, str):
            ch_idx = self.channel_dict[channel]
        else:
            ch_idx = channel
        img = self.get_data().copy()
        img_ch = img[0, 0, ch_idx, :, :, :]
        img[0, 0, ch_idx, :, : :] = img_ch * factor
        self.set_data(img)

    def get_plane(self, channel: Union[int, str], slice: int) -> np.ndarray:
        if isinstance(channel, str):
            ch_idx = self.channel_dict[channel]
            LOGGING.debug(f"in-memory get_plane: {channel} -> {ch_idx} {slice}")
            return self.data[0, 0, ch_idx, slice, :, :]
        else:
            LOGGING.debug(f"in-memory get_plane: {channel} {slice}")
            return self.data[0, 0, channel, slice, np.newaxis, :, :]

    def set_data(self, data):
        self.data = data

    def set_img(self, img):
        self.img = img

    def get_data(self):
        return self.data

    def get_meta(self):
        return self.img.metadata

    # def quit(self):
    #     self.img = None
    #     self.data = None
    #     self.path = None
    #     self.name = None
    #     self.channel_labels = None

    @staticmethod
    def read_img(path: Path, options: Dict) -> AICSImage:
        img = AICSImage(path)
        if not img.metadata:
            print("Metadata not found in input image")
            # might be a case-by-basis
            # img = AICSImage(path), known_dims="CYX")

        return img

    def read_data(self, options):
        data = self.img.data
        dims = data.shape

        # Haoran: hot fix for 5 dims 3D IMC images
        # if len(self.img.data.shape) == 5:
        #     data = self.img.data[:, :, :, :, :]
        #     dims = data.shape
        #     s, c, z, y, x = dims[0], dims[1], dims[2], dims[3], dims[4]
        # else:
        #     data = self.img.data[:, :, :, :, :, :]
        #     dims = data.shape
        #     s, t, c, z, y, x = dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
        #     if t > 1:
        #         data = data.reshape((s, 1, t * c, z, y, x))

        # older version of aicsimageio<3.2
        if len(dims) == 6:
            s, t, c, z, y, x = dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
            if t > 1:
                data = data.reshape((s, 1, t * c, z, y, x))

        # newer version of aicsimageio>4.0 -> correct dimensions
        elif len(dims) == 5:
            data = data[np.newaxis, ...]

        else:
            print(
                "image/expressions dimensions are incompatible. Please check that its in correct format."
            )

        # convert data to a float for normalization downstream
        # data = data.astype("float")
        # assert data.dtype == "float"
        data = data.astype(np.float32)
        assert data.dtype == np.float32

        return data

    def read_channel_names(self):
        img: AICSImage = self.img
        # cn = get_channel_names(img)
        cn = img.channel_names
        print("Channel names:")
        print(cn)

        return cn

    def get_name(self):
        return self.name

    def get_channel_labels(self):
        return self.channel_labels

    def set_channel_labels(self, channel_list):
        self.channel_labels = channel_list


class MaskStruct(IMGstruct):
    """
    Main structure for segmentation information
    """

    def __init__(self, path: Path, options):
        super().__init__(path, options)
        self.bestz = self.get_bestz()
        self.interior_cells = []
        self.edge_cells = []
        self.cell_index = []
        self.bad_cells = set()
        self.ROI = []

    def read_channel_names(self):
        img: AICSImage = self.img
        # cn = get_channel_names(img)
        cn = img.channel_names
        print("Channel names:")
        print(cn)

        # hot fix to channel names expected
        expected_names = ["cell", "nuclei", "cell_boundaries", "nucleus_boundaries"]

        for i in range(len(cn)):
            cn[i] = expected_names[i]

        return cn

    def get_labels(self, label):
        return self.channel_labels.index(label)

    def set_bestz(self, z):
        self.bestz = z

    def get_bestz(self):
        return self.bestz

    def read_data(self, options):
        bestz = []
        data = self.img.data
        dims = data.shape
        # s,t,c,z,y,x = dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]

        # aicsimageio > 4.0
        if len(dims) == 5:
            data = data[np.newaxis, ...]

        check = data[:, :, :, 0, :, :]
        check_sum = np.sum(check)
        if (
            check_sum == 0 and options.get("image_dimensions") == "2D"
        ):  # assumes the best z is not the first slice
            print("Duplicating best z to all z dimensions...")
            for i in range(0, data.shape[3]):
                x = data[:, :, :, i, :, :]
                y = np.sum(x)
                #                print(x)
                #                print(y)
                if y > 0:
                    bestz.append(i)
                    break
                else:
                    continue

            if options.get("debug"):
                print("Best z dimension found: ", bestz)
            # data now contains just the submatrix that has nonzero values
            # add back the z dimension
            data = x[:, :, :, np.newaxis, :, :]
            # print(data.shape)
            # and replicate it
            data = np.repeat(data, dims[3], axis=3)
            # print(data.shape)
            # set bestz
        else:  # 3D case or that slice 0 is best
            if dims[2] > 1:
                bestz.append(int(data.shape[3] / 2))
            else:
                bestz.append(0)

        # set bestz
        self.set_bestz(bestz)

        # check to make sure is int64
        # data = data.astype(int)
        # assert data.dtype == "int"
        data = data.astype(np.int32)
        assert data.dtype == np.int32

        return data

    def add_edge_cell(self, ncell):
        self.edge_cells.append(ncell)

    def set_edge_cells(self, listofcells):
        self.edge_cells = listofcells

    def get_edge_cells(self):
        return self.edge_cells

    def set_interior_cells(self, listofincells):
        self.interior_cells = listofincells

    def get_interior_cells(self):
        return self.interior_cells

    def set_cell_index(self, cellindex):
        self.cell_index = cellindex

    def get_cell_index(self):
        return self.cell_index

    def set_bad_cells(self, bcells):
        self.bad_cells = bcells

    def get_bad_cells(self):
        return self.bad_cells

    def add_bad_cell(self, cell_index):
        self.bad_cells.add(cell_index)

    def add_bad_cells(self, cell_indexes):
        self.bad_cells.update(cell_indexes)

    def set_ROI(self, ROI):
        self.ROI = ROI

    def get_ROI(self):
        return self.ROI

    # def quit(self):
    #     self.img = None
    #     self.data = None
    #     self.path = None
    #     self.name = None
    #     self.channel_labels = None
    #     self.bestz = None
    #     self.interior_cells = None
    #     self.edge_cells = None
    #     self.bad_cells = None
    #     self.cell_index = None
    #     self.ROI = None


class DiskIMGstruct(IMGstruct):

    def __init__(self, path:Path, options):
        self.img = self.read_img(path, options)
        self.data = None
        self.path = path
        self.name = path.name
        self.channel_labels = self.read_channel_names()
        self.channel_dict = {name: idx for idx, name in enumerate(self.channel_labels)}
        self.cached_data = {}
        self.scale_factors = np.ones((self.img.dims.C))
        LOGGER.debug(f"dask image data: {self.img.xarray_dask_data}")
        LOGGER.debug(f"dask image chunk_size: {self.img.xarray_dask_data.chunksizes}")
        LOGGER.debug(f"scale_factors: {self.scale_factors}")

    def get_data(self):
        raise(NotImplementedError("Disk-based images are a work in progress!"))

    def get_plane(self, channel: Union[int, str], slice: int) -> np.ndarray:
        if isinstance(channel, str):
            ch_idx = self.channel_dict[channel]
            LOGGER.debug(f"Accessing {channel} -> {ch_idx} {slice} from dims {self.img.dims}")
            return self.scale_factors[ch_idx] * self.img.get_image_dask_data("ZYX", T=0, C=ch_idx, Z=[slice]).compute().astype(np.float32)
        else:
            ch_name = self.img.channel_names[channel]
            LOGGER.debug(f"Accessing {channel} ({ch_name}) {slice} dims {self.img.dims}")
            return (self.scale_factors[channel]
                    * self.img.get_image_dask_data(
                        "ZYX", T=0, C=channel, Z=[slice]
                    ).compute().astype(np.float32))

    def apply_scale(self, channel: Union[int, str], factor: float) -> None:
        if factor == 1.0:
            return  # because the scaling has no effect
        if isinstance(channel, str):
            ch_idx = self.channel_dict[channel]
        else:
            ch_idx = channel
        LOGGER.debug(f"scaling channel {ch_idx} by {factor}")
        self.scale_factors[ch_idx] *= factor

