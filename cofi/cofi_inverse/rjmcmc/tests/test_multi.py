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
pd = 1.0

inverser = ReversibleJumpMCMC(x, y, error)
inverser.solve(burnin, total, max_order, None, True, pd, max_partitions)
results = inverser.results
xc = results.x()
meancurve = results.mean()
# print(xc)

# Retrieve the partition location and count information
partlocation = results.partition_location_histogram()
partcount = results.partition_histogram()

print(len(results.partitions()))
print("---")
print(len(results.partition_histogram()))

# Plot the data with black crosses and the mean with a red line
fig = matplotlib.pyplot.figure(1, figsize=(6,7))
a = matplotlib.pyplot.subplot(311)
a.plot(x, y, 'ko', xc, meancurve, 'r-')
a.set_xlim(xmin, xmax)
a.set_ylabel("data and mean curve")

b = matplotlib.pyplot.subplot(312)
b.bar(xc, partlocation, xc[1] - xc[0])
b.set_xlim(xmin, xmax)
b.set_ylabel("partition locations")

c = matplotlib.pyplot.subplot(313)
c.hist(partcount, bins=5, range=(0, 5), align='left')
c.set_ylabel("#partitions")

matplotlib.pyplot.show()


