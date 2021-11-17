from .model_params import Model, Parameter


class BaseForward:
    """Base class for all forward solvers in CoFI.

    All forward solvers must be sub-classes of this class and implements two methods:
    1. __init__
    2. misfit()
    """

    def __init__(self, func):
        self.objective = func

    def misfit(self, model: Model):
        return self.objective(*[p.value for p in model.params])

class DataMisfitForward(BaseForward):
    """
    General class holder for objective functions that are calculated from data misfit

    feed the data into constructor, and specify a misfit function
    """

    def __init__(self, data, distance):
        # distance can be a function or a string
        self.data = data
        
        if isinstance(distance, function):
            self.distance = distance
        elif isinstance(distance, str):
            self.distance_name = distance
            # TODO - define the actual distance functions
            # if distance == 'l2':
            #     pass
            # else:
            #     pass

        # TODO self.objective = ???

    def misfit(self, model: Model):
        if self.distance_name:
            raise NotImplementedError("distance functions specified by str not implemented yet")
        
        return super().misfit(model)            
