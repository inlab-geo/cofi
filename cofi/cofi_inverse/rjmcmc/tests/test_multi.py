from os import error
from cofi.cofi_inverse.rjmcmc import ReversibleJumpMCMC
import matplotlib
import matplotlib.pyplot

# --------------------- load data ---------------------
f = open("./test2_data.txt", "r")
lines = f.readlines()

x = []
y = []

for line in lines:
    columns = line.split()

    x.append(float(columns[0]))
    y.append(float(columns[1]))

f.close()

# ---------------------- analysis ---------------------
xmin = 0.0
xmax = 10.0
sigma = 3.0
error = [sigma] * len(x)

burnin = 10000
total = 50000
max_partitions = 10
max_order = 1

inverser = ReversibleJumpMCMC(x, y, error, True)
