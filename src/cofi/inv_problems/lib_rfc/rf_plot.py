import numpy as np
import matplotlib.pyplot as plt
time1, RFo = np.loadtxt("RF_obs.dat", unpack=True)
time2, RFp = np.loadtxt("RF.out", unpack=True)
plt.title(" Observed and predicted receiver functions")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.plot(time1, RFo, 'k-', label='Observed')
plt.plot(time2, RFp, 'r-', label='Predicted')
plt.legend()
plt.show()
plt.savefig('RF_plot.pdf',format='PDF')
