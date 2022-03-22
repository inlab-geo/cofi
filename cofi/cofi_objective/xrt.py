from .. import Model, BaseForward, BaseObjective
from .lib_xrayTomography import tracer, displayModel

import numpy as np


class XRayTomographyObjective(BaseObjective):
    def __init__(
        self,
        data_src_intensity=None,
        data_rec_intensity=None,
        data_paths=None,
        data_attns=None,
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
        """Constructor for XRayTomographyObjective

        When initialising an instance of XRayTomographyObjective, make sure to
        specify in one of the following 3 forms:
        - data_src_intensity, data_rec_intensity, data_paths
        - data_paths, data_attns
        - data_file

        The other arguments, including data_attributes, n_x, n_y and extent, are
        optional (or have default values as described below). extent can be
        important to specify when your dataset is not big enough to cover the
        whole range.

        :param data_src_intensity: [description], defaults to None
        :type data_src_intensity: [type], optional
        :param data_rec_intensity: [description], defaults to None
        :type data_rec_intensity: [type], optional
        :param data_paths: [description], defaults to None
        :type data_paths: [type], optional
        :param data_attns: [description], defaults to None
        :type data_attns: [type], optional
        :param data_file: [description], defaults to None
        :type data_file: [type], optional
        :param data_attributes: [description], defaults to { "src_x": 0, "src_y": 1, "src_intensity": 2, "rec_x": 3, "rec_y": 4, "rec_intensity": 5, }
        :type data_attributes: dict, optional
        :param n_x: [description], defaults to 50
        :type n_x: int, optional
        :param n_y: [description], defaults to 50
        :type n_y: int, optional
        :param extent: [description], defaults to None
        :type extent: [type], optional
        :raises ValueError: [description]
        :raises ValueError: [description]
        :raises ValueError: [description]
        :raises ValueError: [description]
        :raises ValueError: [description]
        :raises ValueError: [description]
        """
        if isinstance(data_src_intensity, str):
            data_file = data_src_intensity
            data_src_intensity = None
        if data_src_intensity is None and data_attns is None:  # data from file path
            # ####### INPUT VALIDATION #######
            if data_file is None:
                raise ValueError(
                    "Please provide observed data while initialising"
                    " XRayTomographyObjective"
                )
            try:
                dataset = np.loadtxt(data_file)
            except Exception:
                raise ValueError(
                    "Please provide a valid file path while initialising"
                    " XRayTomographyObjective"
                )
            # ####### END VALIDATION #######
            data_src_intensity = dataset[:, data_attributes["src_intensity"]]
            data_rec_intensity = dataset[:, data_attributes["rec_intensity"]]
            self.paths = np.zeros([dataset.shape[0], 4])
            self.paths[:, 0] = dataset[:, data_attributes["src_x"]]
            self.paths[:, 1] = dataset[:, data_attributes["src_y"]]
            self.paths[:, 2] = dataset[:, data_attributes["rec_x"]]
            self.paths[:, 3] = dataset[:, data_attributes["rec_y"]]
        elif (
            data_src_intensity is not None
        ):  # <- data as arguments passed in (source/receiver locations + paths)
            # ####### INPUT VALIDATION #######
            if data_rec_intensity is None or data_paths is None:
                raise ValueError(
                    "Please provide full data while initialising"
                    " XRayTomographyObjective, including data_src_intensity,"
                    " data_rec_intensity and data_paths"
                )
            if (data_src_intensity.shape[0] != data_rec_intensity.shape[0]) or (
                data_src_intensity.shape[0] != data_paths.shape[0]
            ):
                raise ValueError(
                    "The dimensions between data_src_intensity, data_rec_intensity and"
                    " data_paths don't match; you need to provide them with the same"
                    " rows count"
                )
            if data_paths.shape[1] != 4:
                raise ValueError(
                    "The given data_paths should have exactly 4 columns that refer to"
                    " source and receiver locations in forms of x and y coordinates;"
                    f" instead we got data_paths of shape {data_paths.shape}"
                )
            # ####### END VALIDATION #######
            self.paths = data_paths
        else:  # <- data as arguments passed in (d as attenuation coefficients + paths)
            # ####### INPUT VALIDATION #######
            if data_paths is None:
                raise ValueError(
                    "Please provide full data while initialising"
                    " XRayTomographyObjective, including data_attns and data_paths"
                )
            if data_attns.shape[0] != data_paths.shape[0]:
                raise ValueError(
                    "The dimensions between data_attns and data_paths don't match; "
                    "you need to provide them with the same rows count"
                )
            if data_paths.shape[1] != 4:
                raise ValueError(
                    "The given data_paths should have exactly 4 columns that refer to"
                    " source and receiver locations in forms of x and y coordinates;"
                    f" instead we got data_paths of shape {data_paths.shape}"
                )
            # ####### END VALIDATION #######
            self.paths = data_paths
            self.d = data_attns

        if not hasattr(self, "d"):
            self.d = -np.log(data_rec_intensity) + np.log(data_src_intensity)
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
        else:  # given extent, check bounds from paths data
            # ####### INPUT VALIDATION #######
            if (
                extent[0] > inferred_extent[0]
                or extent[1] < inferred_extent[1]
                or extent[2] > inferred_extent[2]
                or extent[3] < inferred_extent[3]
            ):
                raise ValueError(
                    "The paths data is out of bounds based on provided region's"
                    f" extent; inferred region's extent: {inferred_extent}"
                )
            # ####### END VALIDATION #######
            self.extent = extent

    def misfit(self, model):
        model = model.values() if isinstance(model, Model) else np.asanyarray(model)
        if len(model.shape) == 1:
            try:
                model = model.reshape([self.n_x, self.n_y])
            except:
                raise ValueError(
                    f"You've provided model if shape: {model.shape}, however we expect"
                    f" one in shape ({self.n_x}, {self.n_y}); alternatively, reset the"
                    " grid dimensions using method 'set_grid_dimensions(n_x, n_y)' on"
                    " objective instance and try again."
                )
        else:
            if model.shape[0] != self.n_x or model.shape[1] != self.n_y:
                raise ValueError(
                    f"You've provided model if shape: {model.shape}, however we expect"
                    f" one in shape ({self.n_x}, {self.n_y}); alternatively, reset the"
                    " grid dimensions using method 'set_grid_dimensions(n_x, n_y)' on"
                    " objective instance and try again."
                )
        d_estimated = self.fwd.calc(model, self.paths, self.extent)
        return np.linalg.norm(self.d - d_estimated)

    def set_grid_dimensions(self, n_x, n_y):
        self.n_x = n_x
        self.n_y = n_y

    def initial_model(self):
        return np.random.rand(self.n_x, self.n_y)

    def basis_matrix(self):
        return self.fwd.basis_function(self.paths, self.n_x, self.n_y, self.extent)

    def data_x(self):
        return self.basis_matrix()

    def data_y(self):
        return self.d

    def residual(self, model):
        model = model.reshape([self.n_x, self.n_y])
        d_estimated = self.fwd.calc(model, self.paths, self.extent)
        return self.d - d_estimated

    def gradient(self, model):
        return np.squeeze(self.jacobian(model).T @ self.residual(model))

    def hessian(self, model):
        g = self.basis_matrix()
        return g.T @ g

    def jacobian(self, model):
        return self.basis_matrix()

    def display(
        self, model, paths=None, extent=None, clim=None, cmap=None, figsize=(6, 6)
    ):
        if isinstance(model, Model):
            model = model.values()
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

    def basis_function(self, paths, n_x, n_y, extent=(0, 1, 0, 1)):
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
