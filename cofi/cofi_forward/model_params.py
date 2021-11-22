import numpy as np
from numbers import Number
from dataclasses import dataclass
from typing import List, Union
from scipy import stats
import yaml


@dataclass
class Parameter:
    """ general class for holding a CoFI model parameter """

    name: str
    value: Union[Number, np.ndarray] = None
    pdf: Union[stats.rv_continuous, np.ndarray] = None
    # continuous distributions in https://docs.scipy.org/doc/scipy/reference/stats.html

    def __post_init__(self):
        if self.value is None and self.pdf is None:
            raise ValueError(
                f"Specified parameter {self.name} has no initial value AND no distribution. You must either specify a value or a range/distribution for each parameter"
            )

        # if pdfs are specified, check they are done correctly.
        if self.pdf is not None:
            if self.value is not None:
                if isinstance(self.value, Number):
                    if not isinstance(self.pdf, stats.rv_continuous):
                        raise ValueError(
                            f"Specified PDF for parameter {self.name} id not a continuous distribution! It is instead a {type(self.pdf)} which is not allowed"
                        )
                    if self.pdf.pdf(self.value) == 0.0:
                        raise ValueError(
                            f"Initial value {self.value} for parameter {self.name} has zero density in specified pdf"
                        )
                elif isinstance(self.value, np.ndarray):
                    # so pdf should be same shape as value and should be all pdfs
                    if not isinstance(self.pdf, np.ndarray):
                        raise ValueError(
                            f"Specified PDF for parameter {self.name} must be an array of PDFs"
                        )
                    elif self.pdf.shape != self.value.shape:
                        raise ValueError(
                            f"Specified PDF for parameter {self.name} must be an array of PDFs with same shape as {self.name}, but {self.name} was shape {self.value.shape} and pdf was shape {self.pdf.shape}"
                        )
                    # OK, so its an array of the right shape. Check the type
                    if False in [
                        isinstance(item, stats.rv_continuous)
                        for item in self.pdf.ravel()
                    ]:
                        raise ValueError(
                            f"Specified PDF for parameter {self.name} must be an array of PDFs"
                        )
                    for i, v in enumerate(self.value.ravel()):
                        if self.pdf.ravel()[i].pdf(v) == 0.0:
                            raise ValueError(
                                f"Initial value at index {i} for parameter {self.name} has zero density in specified pdf"
                            )
            else:  # value is None, so we need to initialize it from pdf
                if isinstance(self.pdf, stats.rv_continuous):
                    self.value = self.pdf.rvs()
                elif isinstance(self.pdf, np.ndarray):
                    self.value = np.array(
                        [item.rvs() for item in self.pdf.ravel()]
                    ).reshape(self.pdf.shape)
                else:
                    raise ValueError(
                        f"specified PDF not of expected type. Expected rv_continuous or array of rv_continuous"
                    )
        else:  # PDF is None, but value is specified. This is fine, we dont need to do anything
            pass

    def __repr__(self) -> str:
        return yaml.safe_dump(self.asdict())

    # utility method to convert this to a dictionary that can be turned into a dictionary, for writing to yaml
    def asdict(self) -> dict:
        res = dict(name=self.name)
        if self.value is not None:
            if isinstance(self.value, np.ndarray):
                res["value"] = self.value.tolist()
            else:
                res["value"] = self.value
        if self.pdf is not None:
            if isinstance(self.pdf, stats.rv_continuous):
                res["pdf"] = f"{self.pdf.dist.name} {' '.join(map(str, self.pdf.args))}"
            else:
                pdfa = np.empty(self.pdf.shape, dtype=object).flatten()
                for i, item in enumerate(self.pdf.ravel()):
                    pdfa[i] = f"{item.dist.name} {' '.join(map(str, item.args))}"
                res["pdf"] = pdfa.reshape(self.pdf.shape).tolist()
        return res


@dataclass
class Model:
    """ general class for holding a CoFI model """

    def __init__(self, **kwargs):
        self.params = []

        for nm, item in kwargs.items():
            if not isinstance(nm, str):
                raise ValueError(
                    f"Invalid argument to Model(): expected a list of name,value tuples, but first element of one was not a string: {nm}"
                )
            if isinstance(item, tuple):
                val, pdf = item
            else:
                val, pdf = item, None
            self.params.append(Parameter(name=nm, value=val, pdf=pdf))

    def values(self) -> np.array:
        return np.array([p.value for p in self.params])

    def length(self) -> int:
        return len(self.params)

    def to_yamlizable(self):
        return [p.asdict() for p in self.params]

    @staticmethod
    def init_from_yaml(yamldict: dict):
        if "parameters" not in yamldict:
            raise Exception(
                f"Model specification in YML file *must* contain 'parameters' information for your model"
            )

        # parameters should be a list of dictionaries
        if not isinstance(yamldict["parameters"], list):
            raise ValueError(
                f"In your YML file, you must specify 'parameters' for your model as a list"
            )
        args = {}
        for p in yamldict["parameters"]:
            if not isinstance(p, dict):
                raise ValueError(
                    f"each paramater in model in YML file must be (key, value) pairs"
                )
            if "name" not in p or not isinstance(p["name"], str):
                raise ValueError(f"Each parameter must have a name")
            nm = p["name"]
            val = p["value"] if "value" in p else None
            pdf = p["bounds"] if "bounds" in p else None

            def parsepdf(toparse: str) -> stats.rv_continuous:
                bits = toparse.split()
                if bits[0] not in dir(stats):
                    raise ValueError(f"Unknown distribution specified: {bits[0]}")
                pdfstr = f"stats.{bits[0]}({','.join(bits[1:])})"
                return eval(pdfstr)

            if pdf is not None:
                if isinstance(pdf, list):
                    pdf = np.asarray(pdf)
                    pdf = np.array([parsepdf(item) for item in pdf.ravel()]).reshape(
                        pdf.shape
                    )
                else:
                    pdf = parsepdf(pdf)
            args[nm] = (val, pdf)
        return Model(**args)
