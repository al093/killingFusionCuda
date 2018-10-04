from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import sys

#default values
fileU = sys.argv[1]
fileV = sys.argv[2]
fileW = sys.argv[3]
fileWeights = sys.argv[4]
zSlice = 40
dim = 80;

# if len(sys.argv) is 3: #use dimension ans slice
    # zSlice = sys.argv[1]
    # dim = sys.argv[2]
# elif len(sys.argv) is 5: #all u,v,w files and the provided, use those paths
    # zSlice = sys.argv[1]
    # dim = sys.argv[2]
    # fileU = sys.argv[3]
    # fileV = sys.argv[4]
    # fileW = sys.argv[5]
# else:
    # raise ValueError('The number of arguments provied is incorrect, please provide the z slice and three files for u, v, w deformation field')

# read the u, v, w into a np arrays.
#reshaping assumes that the the array was stored in row major order.
u = np.loadtxt(fileU)
u = u.reshape((dim,dim))

v = np.loadtxt(fileV)
v = v.reshape((dim,dim))

w = np.loadtxt(fileW)
w = w.reshape((dim,dim))

weights = np.loadtxt(fileWeights)
# Make the grid
x, y = np.meshgrid(np.arange(0, dim, 1),
                      np.arange(dim , 0, -1))

# Color
occurrence = weights
norm = colors.Normalize()
norm.autoscale(occurrence)
cm1 = cm.copper

sm = cm.ScalarMappable(cmap=cm1, norm=norm)
sm.set_array([])

plt.figure()
plt.title(sys.argv[5])
q = plt.quiver(x, y, u, v, units='xy', scale=0.05, angles='xy', scale_units='xy', color=cm1(norm(weights)))
plt.colorbar(sm)
imageName = str(sys.argv[5].replace(".txt", "")) + ".jpg"
plt.savefig(imageName)
plt.show()