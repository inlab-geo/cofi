from os import error
from cofi.samplers import ReversibleJumpMCMC
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

# ---------------------- partition analysis ---------------------
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

# Plot the data with black crosses and the mean with a red line
# fig = matplotlib.pyplot.figure(1, figsize=(6,7))
# a = matplotlib.pyplot.subplot(311)
# a.plot(x, y, 'ko', xc, meancurve, 'r-')
# a.set_xlim(xmin, xmax)
# a.set_ylabel("data and mean curve")

# b = matplotlib.pyplot.subplot(312)
# b.bar(xc, partlocation, xc[1] - xc[0])
# b.set_xlim(xmin, xmax)
# b.set_ylabel("partition locations")

# c = matplotlib.pyplot.subplot(313)
# c.hist(partcount, bins=5, range=(0, 5), align='left')
# c.set_ylabel("#partitions")

# matplotlib.pyplot.show()


# ---------------------- order analysis ---------------------
inverser_ord = ReversibleJumpMCMC(x, y, error)
inverser_ord.solve(multi_partition=True, partition_move_std=pd)
xc_ord = inverser_ord.results.x()
meancurve_ord = inverser_ord.results.mean()
partlocation_ord = inverser_ord.results.partition_location_histogram()
partcount_ord = inverser_ord.results.partitions()

# fig = matplotlib.pyplot.figure(1)
# a = matplotlib.pyplot.subplot(211)
# a.plot(x, y, 'ko', xc_ord, meancurve_ord, 'r-')
# a.set_xlim(xmin, xmax)
# b = matplotlib.pyplot.subplot(212)
# b.bar(xc_ord, partlocation_ord, xc_ord[1]-xc_ord[0])
# b.set_xlim(xmin, xmax)

# fig = matplotlib.pyplot.figure(2)
# a = matplotlib.pyplot.subplot(111)
# a.hist(partcount_ord, bins=5, range=(0, 5), align='left')

# matplotlib.pyplot.show()


# ---------------------- own sampler ---------------------
sample_rate = 250


def sampler(x, y, i):
    return i % sample_rate == 0


burnin = 10000
total = 50000
max_partitions = 10
max_order = 3

inverser_sample = ReversibleJumpMCMC(x, y, error)
inverser_sample.solve(burnin, total, max_order, sampler, True, pd, max_partitions)
sample_x = inverser_sample.sample_x
sample_curves = inverser_sample.sample_curves

fig = matplotlib.pyplot.figure(1)
ax = fig.add_subplot(111)
yc = 0.5
yalpha = 1.0 / ((1.0 - yc) * float(len(sample_curves)))
for sy in sample_curves:
    ax.plot(sample_x, sy, color=str(yc), alpha=yalpha, linestyle="-", linewidth=10)
ax.plot(results.x(), results.mean(), "r-")
ax.plot(x, y, "ko")
ax.set_xlim(xmin, xmax)

fig = matplotlib.pyplot.figure(2)
ax = fig.add_subplot(111)
ax.plot(results.x(), results.mean(), "r-")
ax.plot(x, y, "ko")
ax.plot(results.x(), results.credible_min(), "b:")
ax.plot(results.x(), results.credible_max(), "b:")
ax.set_xlim(xmin, xmax)

matplotlib.pyplot.show()

# ---------------------- not sure about error ---------------------
lambda_min = 0.5
lambda_max = 3.0
lambda_std = 0.1

inverser_lambda = ReversibleJumpMCMC(x, y, error, lambda_min, lambda_max, lambda_std)
inverser_lambda.solve(multi_partition=True, partition_move_std=pd)
xc_lambda = inverser_lambda.results.x()
meancurve_lambda = inverser_lambda.results.mean()

partlocation_lambda = inverser_lambda.results.partition_location_histogram()
partcount_lambda = inverser_lambda.results.partitions()

p = inverser_lambda.results.proposed()
a = inverser_lambda.results.acceptance()
print(a, p)

print("Lambda Acceptance Rate:", float(a[4]) / float(p[4]) * 100.0)

lh = inverser_lambda.results.lambda_history()

fig = matplotlib.pyplot.figure(1)
a = matplotlib.pyplot.subplot(211)
a.plot(x, y, "ko", xc_lambda, meancurve_lambda, "r-")
a.set_xlim(xmin, xmax)

b = matplotlib.pyplot.subplot(212)
b.bar(xc, partlocation_lambda, xc[1] - xc[0])
b.set_xlim(xmin, xmax)

fig = matplotlib.pyplot.figure(2)
matplotlib.pyplot.plot(range(len(lh)), lh)

fig = matplotlib.pyplot.figure(3)
a = matplotlib.pyplot.subplot(111)
lsamples = lh[10000:]
n, bins, patches = a.hist(lsamples, 100, range=(lambda_min, lambda_max))
a.set_title("Histogram of Lambda")
a.set_xlabel("Lambda")
a.set_ylabel("Count")

print("Lambda average:", sum(lsamples) / float(len(lsamples)))

matplotlib.pyplot.show()
