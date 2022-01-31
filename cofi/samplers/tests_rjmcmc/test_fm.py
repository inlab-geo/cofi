from os import error
import sys
from cofi.cofi_objective import BaseObjectiveFunction
from cofi.samplers import ReversibleJumpMCMC
from cofi.cofi_objective import Parameter, Model, BaseForward
import matplotlib
import matplotlib.pyplot
from scipy.stats import norm


# --------------------- load data ---------------------
f = open("test3_data.txt", "r")
lines = f.readlines()

x = []
y = []

for line in lines:
    columns = line.split()

    x.append(float(columns[0]))
    y.append(float(columns[1]))

f.close()


# --------------------- define hyperparameters ---------------------
xmin = 0.0
xmax = 10.0
pd = 1.0  # standard deviation for the move partition
sigma = (
    5.0  # estimate of the noise parameter, which will affect the tightness of the fit
)


# --------------------- define forward model ---------------------

# (min, max, std. deviation) for each parameter.
# local_parameters = [(-50.0, 50.0, 2.0)]

# Global parameters are parameters that are the same in all partitions,
# an example is a dc bias that is constant across the domain. In this
# example we have no global parameters. They are specified in the
# same manner as the local parameters.
# global_parameters = None

parameter = Parameter(name="local", pdf=norm(0, 2))


def my_misfit_partition(*localvalues, boundaries):
    phi = 0
    for (xi, yi) in zip(x, y):
        if (xi < boundaries[0]) or (xi > boundaries[len(boundaries) - 1]):
            # if the point is outside the range then ignore it (shouldn't happen)
            continue

        ci = 0
        while (ci < len(localvalues)) and (xi > boundaries[ci + 1]):
            ci += 1

        if ci == len(localvalues):
            print("error: failed to find cell")
            print(" ", boundaries, xi)
            sys.exit(-1)

        ym = localvalues[ci][0]

        dy = yi - ym
        phi = phi + (dy * dy)

    return phi / (2.0 * sigma * sigma)


class MyObjectiveFunction(BaseObjectiveFunction):
    def __init__(self):
        super().__init__(self, my_misfit_partition)


# --------------------- inverse ---------------------
inverser = ReversibleJumpMCMC(Model(("local", norm(0, 2))), MyObjectiveFunction())

# results = rjmcmc.forwardmodel_part1d(
#     local_parameters, global_parameters, my_loglikelihood, xmin, xmax, pd
# )
