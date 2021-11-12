from os import error
from cofi.cofi_inverse import ReversibleJumpMCMC
import matplotlib
import matplotlib.pyplot


# --------------------- load data ---------------------
f = open('test3_data.txt', 'r')
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
pd = 1.0   # standard deviation for the move partition
sigma = 5.0


# (min, max, std. deviation) for each parameter.
local_parameters = [(-50.0, 50.0, 2.0)]

# Global parameters are parameters that are the same in all partitions,
# an example is a dc bias that is constant across the domain. In this 
# example we have no global parameters. They are specified in the 
# same manner as the local parameters.
global_parameters = None

def my_loglikelihood(globalvalues, boundaries, localvalues):
    global x, y, sigma

    # the parameters to this function are:
    # - a list of the global values
    # - a sorted list of the boundary positions including the start and end position
    # - the values in each region (so len(boundaries) == len(localvalues) + 1)
