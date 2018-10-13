from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import sys
import os

#default values
fileData = sys.argv[1]
fileLevelSet = sys.argv[2]
fileKilling = sys.argv[3]
fileTotal = sys.argv[4]
figureName = sys.argv[5]

saveImg = False
frameNumber = 0
if(len(sys.argv) == 7):
    frameNumber = sys.argv[6]
    saveImg = True;

data = np.loadtxt(fileData)
levelSet = np.loadtxt(fileLevelSet)
killing = np.loadtxt(fileKilling)
total = np.loadtxt(fileTotal)

plt.plot(data[1:], 'r')
plt.plot(levelSet[1:], 'g')
plt.plot(killing[1:], 'b')
plt.plot(total[1:], 'k')

resultsDir = "./bin/result/"

if saveImg:
    print("Saving Energy Plot")
    imageName = resultsDir + str(figureName.replace(".txt", "_")) + str(frameNumber)+ "energy.png"
    plt.savefig(imageName, format='png', dpi=1000)
