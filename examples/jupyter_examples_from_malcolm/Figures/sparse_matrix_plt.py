import numpy as np
import matplotlib.pyplot as plt

img = (np.random.standard_normal([20, 20, 3]) * 255).astype(np.uint8)
img[img>50] = 255
img[img<42] = 255
img[img<50] = 0
plt.imshow(img)
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.show()