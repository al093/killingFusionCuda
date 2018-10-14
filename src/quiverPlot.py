from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np
import sys

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

#default values
fileU = './bin/result/u.txt'
fileV = './bin/result/v.txt'
fileW = './bin/result/w.txt'
zSlice = 40
dim = 80;

if len(sys.argv) is 3: #use dimension ans slice
    zSlice = sys.argv[1]
    dim = sys.argv[2]
elif len(sys.argv) is 5: #all u,v,w files and the provided, use those paths
    zSlice = sys.argv[1]
    dim = sys.argv[2]
    fileU = sys.argv[3]
    fileV = sys.argv[4]
    fileW = sys.argv[5]
# else:
    # raise ValueError('The number of arguments provied is incorrect, please provide the z slice and three files for u, v, w deformation field')

# read the u, v, w into a np arrays.
#reshaping assumes that the the array was stored in row major order.
u = np.loadtxt(fileU)
u = u.reshape((dim,dim,dim))

v = np.loadtxt(fileV)
v = v.reshape((dim,dim,dim))

w = np.loadtxt(fileW)
w = w.reshape((dim,dim,dim))

lengths = np.sqrt(u**2 + v**2 + w**2);
lengths = lengths.ravel().tolist()
norm = Normalize()
norm.autoscale(lengths)
colormap = cm.spectral

# Make the grid
x, y, z = np.meshgrid(np.arange(0, dim, 1),
                      np.arange(0, dim, 1),
                      np.arange(0, dim, 1))


ax.quiver(y, z, x, v, w, u, length=1, color = colormap(norm(lengths)))

#plt.show()