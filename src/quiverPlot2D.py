from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np
import sys

#default values
fileU = './bin/result/u.txt'
fileV = './bin/result/v.txt'
fileW = './bin/result/w.txt'
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

# Make the grid
x, y = np.meshgrid(np.arange(0, dim, 1),
                      np.arange(0, dim, 1))

plt.figure()
plt.title('deformation plot')
q = plt.quiver(x, y, u, v, units='xy', scale=None, angles='xy', scale_units='xy')
plt.show()