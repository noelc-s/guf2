import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def npread(file, sep=',', header=None):
    df = pd.read_csv(file, sep=sep, header=header)
    arr = np.asarray(df)
    return arr

def npwrite(arr, file):
    df = pd.DataFrame(arr)
    df.to_csv(file, header=False, index=False)

def mix(color1, color2, val):
    res = []
    for c1, c2 in zip(color1,color2):
        res.append(c1*(1-val)+c2*(val))
    return res

gray = (0.1,0.1,0,1)
blue = (0,0.5,0.9)
red = (0.9,0.2,0)

sa = []

f0 = npread('ForceCharCL/fz_0.csv')
wb = np.arange(f0.shape[0])/100

fp = []


## Stroke averages for all the individual modes
mag = np.arange(-10,10)
directions = ['fx','fy','fz','mx','my','mz']

for file in range(0,6):
    for direction in range(0,6):
        stroke_avg = []
        for j in range(-10,10):
            f_j = npread('ForceCharCL/'+directions[file]+'_{}.csv'.format(j))[:,direction]
            stroke_avg.append(np.mean(f_j[100:]))
            # fp.append(f_j)
        ax=plt.subplot(2,3,direction+1)
        ax.set_title(directions[direction])
        # for i in range(6):
        #     plt.plot(mag[i:i+2],stroke_avg[i:i+2], '.-', lw=3, ms=10, color=mix(red,gray,file/5))
        # for i in np.arange(11):
            # plt.plot(mag[i:i+2],stroke_avg[i:i+2], '.-', lw=3, ms=10)
        if direction == file:
            plt.plot(mag,stroke_avg,'-',lw=4, ms=5)
        else:
            plt.plot(mag,stroke_avg,'--',lw=2, ms=5)
plt.legend(directions)

# plt.xlabel("Control Magnitude")
# plt.ylabel("Stroke-Averaged $F_z$")
# plt.gca().spines["right"].set_visible(False)
# plt.gca().spines["top"].set_visible(False)
# plt.plot(wb,np.transpose(fp))
plt.show()


#### Cross-directions
# mag = np.arange(-5,5)
# X, Y = np.meshgrid(mag,mag)
# directions = ['fx','fy','fz','mx','my','mz']
# direction1 = 0
# direction2 = 2
# fig = plt.figure()
# for direction in range(0,6):
#     stroke_avg = np.empty((10,10))
#     for j1 in range(-5,5):
#         for j2 in range(-5,5):
#             f_j = npread('ForceCharacterization/'+directions[direction1]+'_'+directions[direction2]+'_'+str(j1)+'_'+str(j2)+'.csv')[:,direction]
#             stroke_avg[j1+5,j2+5] = np.mean(f_j[100:])
#     ax = fig.add_subplot(2, 3, direction+1, projection='3d')
#     ax.set_title(directions[direction])
#     ax.plot_surface(X,Y,stroke_avg)
#     ax.set_xlabel(directions[direction2])
#     ax.set_ylabel(directions[direction1])
# plt.show()