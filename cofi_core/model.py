import numpy as np
from numbers import Number 
from dataclasses import dataclass
from typing import List, Union

INT_TYPES = [np.uint8, np.int8, np.int16, np.int32, np.int64]
FLOAT_TYPES = [np.float16, np.float32, np.float64, np.float]



@dataclass
class Parameter:
    """ general class for holding a CoFI model parameter """
    name: str
    paramtype: str
    value: Union[Number, np.ndarray] = None
    pdf: scipy.stats.rv_continuous = None

    def __post_init__(self):
        if self.value is None and self.pdf is None:
            raise ValueError(f"Specified parameter {name} has no initial value AND no distribution. You must either specify a value or a range/distribution for each parameter")
        elif value is None:
            self.value = self.pdf.rvs() # dram from distribution


@dataclass
class Model:
    """ general class for holding a CoFI model """
    def __init__(self, **kwargs):
        self.params = []

        for nm, val in kwargs.items():
            if not isinstance(nm, str):
                raise ValueError(f"Invalid argument to Model(): expected a list of name,value tuples, but first element of one was not a string: {nm}")
            vtype = str(type(val) if isinstance(val, Number) else val.dtype)

            if isinstance(val, np.ndarray) and len(val.shape) not in [1,2,3]:
                raise ValueError(f"Currently on 1d vectors, 2d matrices, and 3d tensors are supported, but you tried to create a model with parameter shape {val.shape}")
            self.params.append(Parameter(name=nm, paramtype=vtype, value=val))

    @staticmethod
    def init_from_yaml(yamldict: dict):
        if "parameters" not in yamldict:
            raise Exception(f"")







    
