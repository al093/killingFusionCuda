from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import sys
import os

#default values
fileU = sys.argv[1]
fileV = sys.argv[2]
fileW = sys.argv[3]
fileSdf = sys.argv[4]
figureName = sys.argv[5]

saveImg = False
frameNumber = 0
if(len(sys.argv) == 7):
    frameNumber = sys.argv[6]
    saveImg = True;

dim = 80;

u = np.loadtxt(fileU)
v = np.loadtxt(fileV)
w = np.loadtxt(fileW)
sdf = np.loadtxt(fileSdf)

# Make the grid
x, y = np.meshgrid(np.arange(0, dim, 1),
                   np.arange(0, dim, 1))

for i in range(sdf.shape[0]):
	if (sdf[i] > .5) or (sdf[i] < -.5):
		u[i] = 0
		v[i] = 0

u = u.reshape((dim,dim))
v = v.reshape((dim,dim))
w = w.reshape((dim,dim))

norm = colors.Normalize()
norm.autoscale(sdf)
cm1 = cm.inferno

sm = cm.ScalarMappable(cmap=cm1, norm=norm)
sm.set_array([])

plt.figure()
#plt.title(figureName)
q = plt.quiver(x, y, u, v, units='xy', scale=1, angles='xy', scale_units='xy', color=cm1(norm(sdf)))
#plt.colorbar(sm)

ax = plt.gca()
ax.invert_yaxis()

resultsDir = "./bin/result/"

if saveImg:
    print("Saving Quiver Image")
    imageName = resultsDir + str(figureName.replace(".txt", "_")) + str(frameNumber)+ ".png"
    plt.savefig(imageName, format='png', dpi=1000)

# plt.show()
