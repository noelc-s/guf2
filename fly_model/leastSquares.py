import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def npread(file, sep=',', header=None):
    df = pd.read_csv(file, sep=sep, header=header)
    arr = np.asarray(df)
    return arr

sa = []

f0 = npread('ForceCharacterization/fz_0.csv')
wb = np.arange(f0.shape[0])/100

fp = []

mag = np.arange(-10,10)



directions = ['fx','fy','fz','mx','my','mz']

X = np.empty((10*6,6))
Y = np.empty((10*6,6))

i = 0

for file in range(0,6):
    i = 0
    for direction in range(0,6):
        stroke_avg = []
        cmd = []
        for j in range(-5,5):
            f_j = npread('ForceCharacterization/'+directions[file]+'_{}.csv'.format(j))[:,direction]
            stroke_avg.append(np.mean(f_j[100:]))
            cmd_tmp = [0,0,0,0,0,0]
            cmd_tmp[file] = 0.00001*j
            cmd.append(cmd_tmp)
            # fp.append(f_j)
        Y[file*10:(file+1)*10,i] = stroke_avg
        i += 1

    X[file*10:(file+1)*10,:] = cmd

        # plt.plot(mag,stroke_avg,'.-',lw=2, ms=5)

A = np.linalg.lstsq(X,Y,rcond=None)
print(A)

A_inv = np.linalg.inv(A[0])
sol = A_inv*np.array([1,0,0,0,0,0])
print(sol[:,0])

x,y = np.meshgrid(np.arange(0,6),np.arange(0,6))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x,y, A[0])
plt.show()
