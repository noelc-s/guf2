import numpy as np
import numpy.matlib
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

mode_mag_range1 = range(-4,1)
mode_mag_range2 = range(0,5)
mode_mag_range = range(-5,5)
size = len(mode_mag_range)

X = np.empty((size*6,6))
Y = np.empty((size*6,6))

i = 0

for file in range(0,6):
    i = 0
    for direction in range(0,6):
        stroke_avg = []
        cmd = []
        for j in mode_mag_range:
            f_j = npread('ForceCharacterization/'+directions[file]+'_{}.csv'.format(j))[:,direction]
            stroke_avg.append(np.mean(f_j[100:]))
            cmd_tmp = [0,0,0,0,0,0]
            cmd_tmp[file] = 0.00001*j
            cmd.append(cmd_tmp)
            # fp.append(f_j)
        Y[file*size:(file+1)*size,i] = stroke_avg
        i += 1

    X[file*size:(file+1)*size,:] = cmd

        # plt.plot(mag,stroke_avg,'.-',lw=2, ms=5)

Y = Y - np.matlib.repmat(Y[5,:],X.shape[0],1)
X = X*1e5

i = 5

Y = Y[i*size:(i+1)*size,i]
X = X[i*size:(i+1)*size,i]

Y = np.transpose(np.vstack((Y[1:6],Y[5:10])))
X = np.transpose(np.vstack((X[1:6],X[5:10])))

A = np.linalg.lstsq(X,Y,rcond=None)
print(A)

A_inv = np.linalg.inv(A[0])
# sol = A_inv*np.array([1,0,0,0,0,0])*1e-5
# print(sol[:,0])

# x,y = np.meshgrid(np.arange(0,6),np.arange(0,6))
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(x,y, A[0])
# mul = np.dot(A_inv,np.transpose(X[0:size,0]))
#
# plt.plot(mode_mag_range,Y[0:size,0])
# plt.plot(mode_mag_range,mul[0,:])
# plt.show()
mul = np.dot(Y,A_inv)
print(mul)

plt.plot(np.transpose(np.vstack((mode_mag_range1,mode_mag_range2))),Y)
# plt.plot(np.transpose(np.vstack((mode_mag_range1,mode_mag_range2))),np.dot(X,A[0]))
plt.plot(mul,Y)
plt.show()
