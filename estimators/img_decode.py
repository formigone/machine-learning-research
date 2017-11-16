import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('../3x4.png')
print(img.shape)
print(img[0])
print('--')
print(img[1])
print('--')
print(img[2])

img_2 = np.zeros((2, 2, 3))
img_2[0][0] = img[0][0]
img_2[0][1] = img[0][-1]

img_2[1][0] = img[2][0]
img_2[1][1] = img[2][-1]
plt.imsave('2x2.png', img_2)
