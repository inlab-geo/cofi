from cofi.cofi_objective import BaseObjective, BaseForward, Model
from .lib_xrayTomography import tracer, displayModel

import numpy as np


class XRayTomographyObjective(BaseObjective):
    def __init__(
        self,
        data_src_intensity=None,
        data_rec_intensity=None,
        data_paths=None,
        data_file=None,
        data_attributes={
            "src_x": 0,
            "src_y": 1,
            "src_intensity": 2,
            "rec_x": 3,
            "rec_y": 4,
            "rec_intensity": 5,
        },
        n_x=50,
        n_y=50,
        extent=None,
    ):
        if isinstance(data_src_intensity, str):
            data_file = data_src_intensity
            data_src_intensity = None
        if data_src_intensity is None:
            # ####### INPUT VALIDATION #######
            if data_file is None:
                raise ValueError(
                    "Please provide observed data while initialising XRayTomographyObjective"
                )
            try:
                dataset = np.loadtxt(data_file)
            except Exception as e:
                print(e)
                raise ValueError(
                    "Please provide a valid file path while initialising XRayTomographyObjective"
                )
            # ####### END VALIDATION #######
            data_src_intensity = dataset[:, data_attributes["rec_intensity"]]
            data_rec_intensity = dataset[:, data_attributes["src_intensity"]]
            self.paths = np.zeros([dataset.shape[0], 4])
            self.paths[:, 0] = dataset[:, data_attributes["src_x"]]
            self.paths[:, 1] = dataset[:, data_attributes["src_y"]]
            self.paths[:, 2] = dataset[:, data_attributes["rec_x"]]
            self.paths[:, 3] = dataset[:, data_attributes["rec_y"]]
        else:
            # ####### INPUT VALIDATION #######
            if data_rec_intensity is None or data_paths is None:
                raise ValueError(
                    "Please provide full data while initialising XRayTomographyObjective, "
                    "including data_src_intensity, data_rec_intensity and data_paths"
                )
            if (
                data_src_intensity.shape[0] != data_rec_intensity.shape[0]
                or data_src_intensity.shape[0] != data_paths[0]
            ):
                raise ValueError(
                    "The dimensions between data_src_intensity, data_rec_intensity and "
                    "data_paths don't match; you need to provide them with the same rows count"
                )
            if data_src_intensity.shape[1] != 2:
                raise ValueError(
                    f"The given data_src_intensity should have exactly 2 columns that refer "
                    f"to source locations in forms of x and y coordinates; instead we got "
                    f"data_src_intensity of shape {data_src_intensity.shape}"
                )
            if data_rec_intensity.shape[1] != 2:
                raise ValueError(
                    f"The given data_rec_intensity should have exactly 2 columns that refer "
                    f"to source locations in forms of x and y coordinates; instead we got "
                    f"data_rec_intensity of shape {data_rec_intensity.shape}"
                )
            # ####### END VALIDATION #######
            self.paths = data_paths
        self.d = -np.log(data_src_intensity) + np.log(data_rec_intensity)
        self.fwd = XRayTomographyForward()
        self.n_x = n_x
        self.n_y = n_y

        inferred_extent = (
            min(np.min(self.paths[:, 0]), np.min(self.paths[:, 2])),
            max(np.max(self.paths[:, 0]), np.max(self.paths[:, 2])),
            min(np.min(self.paths[:, 1]), np.min(self.paths[:, 3])),
            max(np.max(self.paths[:, 1]), np.max(self.paths[:, 3])),
        )
        if extent is None:
            self.extent = inferred_extent
        else:
            # ####### INPUT VALIDATION #######
            if (
                extent[0] > inferred_extent[0]
                or extent[1] < inferred_extent[1]
                or extent[2] > inferred_extent[2]
                or extent[3] < inferred_extent[3]
            ):
                raise ValueError(
                    f"The paths data is out of bounds based on provided region's extent; "
                    f"inferred region's extent: {inferred_extent}"
                )
            # ####### END VALIDATION #######
            self.extent = extent

    def misfit(self, model):
        pass

    def set_grid_dimensions(self, n_x, n_y):
        self.n_x = n_x
        self.n_y = n_y

    def design_matrix(self):
        return self.fwd.design_matrix(self.paths, self.n_x, self.n_y, self.extent)

    def data_y(self):
        return self.d

    def display(
        self, model, paths=None, extent=None, clim=None, cmap=None, figsize=(6,6)
    ):
        if len(model.shape) == 1:
            model = model.reshape([self.n_x, self.n_y])
        self.fwd.display(
            model,
            paths,
            self.extent if extent is None else extent,
            clim,
            cmap,
            figsize,
        )


class XRayTomographyForward(BaseForward):
    def __init__(self):
        pass

    def calc(self, model, paths, extent=(0, 1, 0, 1)):
        """Perform the forward operation to the X-Ray Tomography problem.

        :param model: the discretized version of the position-dependent attenuation coefficient.
            This is expressed as an array of dimension :math:`(N_x, N_y)`, where :math:`N_x` 
            and :math:`N_y` are the number of cells in :math:`x`-direction and :math:`y`-direction
            respectively.
        :type model: Union[cofi.cofi_objective.Model, np.ndarray, list]
        :param paths: an array of source and receiver locations. This has dimension 
            :math:`(N_{paths}, 4)` so that
            - ``paths[i,0]`` - :math:`x`-location of source for path :math:`i`
            - ``paths[i,1]`` - :math:`y`-location of source for path :math:`i`
            - ``paths[i,2]`` - :math:`x`-location of receiver for path :math:`i`
            - ``paths[i,3]`` - :math:`y`-location of receiver for path :math:`i`
        :type paths: Union[np.ndarray, list]
        :param extent: the model region :math:`(xmin,xmax,ymin,ymax)`, defaults to (0,1,0,1).
            Note that all sources and receivers must lie within, or on the boundary of, this model
            region.
        :type extent: tuple, optional
        :return: the attenuation for each path. It is an array of dimension :math:`(N_{paths})`, 
            with the :math:`i`-th element being :math:`{-\\log{\\frac{I_{rec}^{(i)}}{I_{src}^{(i)}}}}`.
        :rtype: np.ndarray
        """
        model = model.values() if isinstance(model, Model) else np.asanyarray(model)
        self.paths = np.asanyarray(paths)
        attns, self.A = tracer(model, self.paths, extent)
        return attns

    def design_matrix(self, paths, n_x, n_y, extent=(0, 1, 0, 1)):
        paths = np.asanyarray(paths)
        return tracer(np.ones([n_x, n_y]), paths, extent)[1]

    def display(
        self,
        model,
        paths=None,
        extent=(0, 1, 0, 1),
        clim=None,
        cmap=None,
        figsize=(6, 6),
    ):
        displayModel(model, paths, extent, clim, cmap, figsize)
