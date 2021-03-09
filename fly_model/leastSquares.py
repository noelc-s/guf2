import cvxpy as cp
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
            f_j = npread('ForceCharCL/'+directions[file]+'_{}.csv'.format(j))[:,direction]
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

i = 0
j = 2
k = 4

Y11 = np.array([Y[i*size:(i+1)*size,i]]).T
Y12 = np.array([Y[i*size:(i+1)*size,j]]).T
Y13 = np.array([Y[i*size:(i+1)*size,k]]).T
X11 = np.array([X[i*size:(i+1)*size,i]]).T
Y22 = np.array([Y[j*size:(j+1)*size,j]]).T
Y21 = np.array([Y[j*size:(j+1)*size,i]]).T
Y23 = np.array([Y[j*size:(j+1)*size,k]]).T
X22 = np.array([X[j*size:(j+1)*size,j]]).T
Y33 = np.array([Y[k*size:(k+1)*size,k]]).T
Y32 = np.array([Y[k*size:(k+1)*size,j]]).T
Y31 = np.array([Y[k*size:(k+1)*size,i]]).T
X33 = np.array([X[k*size:(k+1)*size,k]]).T

# Y = np.block([[Y11[1:6,:], np.zeros((5,3))],[np.zeros((5,1)), Y11[5:10,:],np.zeros((5,2))],
#               [np.zeros((5,2)), Y22[1:6,:], np.zeros((5,1))],[np.zeros((5,3)),Y22[5:10]]])
# X = np.block([[X11[1:6,:], np.zeros((5,3))],[np.zeros((5,1)), X11[5:10,:],np.zeros((5,2))],
#               [np.zeros((5,2)), X22[1:6,:], np.zeros((5,1))],[np.zeros((5,3)),X22[5:10]]])

# Y = np.block([[Y11[1:6,:], np.zeros((5,1)),Y12[1:6,:], np.zeros((5,1))],
#               [np.zeros((5,1)), Y11[5:10,:],np.zeros((5,1)), Y12[5:10,:]],
#               [Y21[1:6,:], np.zeros((5,1)), Y22[1:6,:], np.zeros((5,1))],
#               [np.zeros((5,1)), Y21[5:10,:],np.zeros((5,1)), Y22[5:10,:]]])
# X = np.block([[X11[1:6,:], np.zeros((5,3))],[np.zeros((5,1)), X11[5:10,:],np.zeros((5,2))],
#               [np.zeros((5,2)), X22[1:6,:], np.zeros((5,1))],[np.zeros((5,3)),X22[5:10]]])

Y = np.block([[Y11[1:6,:], np.zeros((5,1)),Y12[1:6,:], np.zeros((5,1)),Y13[1:6,:], np.zeros((5,1))],
              [np.zeros((5,1)), Y11[5:10,:],np.zeros((5,1)), Y12[5:10,:],np.zeros((5,1)), Y13[5:10,:]],
              [Y21[1:6,:], np.zeros((5,1)), Y22[1:6,:], np.zeros((5,1)), Y23[1:6,:], np.zeros((5,1))],
              [np.zeros((5,1)), Y21[5:10,:],np.zeros((5,1)), Y22[5:10,:],np.zeros((5,1)), Y23[5:10,:]],
             [Y31[1:6,:], np.zeros((5,1)), Y32[1:6,:], np.zeros((5,1)), Y33[1:6,:], np.zeros((5,1))],
             [np.zeros((5,1)), Y31[5:10,:],np.zeros((5,1)), Y32[5:10,:],np.zeros((5,1)), Y33[5:10,:]]])
X = np.block([[X11[1:6,:], np.zeros((5,5))],[np.zeros((5,1)), X11[5:10,:],np.zeros((5,4))],
              [np.zeros((5,2)), X22[1:6,:], np.zeros((5,3))],[np.zeros((5,3)),X22[5:10], np.zeros((5,2))],
              [np.zeros((5,4)), X33[1:6,:], np.zeros((5,1))],[np.zeros((5,5)),X33[5:10]]])


# X = np.block([[X1[1:6,:], np.zeros((5,3))],[np.zeros((5,3)), X1[5:10,:]]])

A = np.linalg.lstsq(X,Y,rcond=None)
print(A)

A = A[0].T
b = [0,1,0,0,0,0]

x = cp.Variable(A.shape[1])

objective = cp.Minimize(0.5 * cp.sum_squares(A@x-b))
constraints = [x >= 0]

prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.ECOS)

# sol = np.linalg.lstsq(A,b,rcond=None)
sol = x.value
print(sol*1e-5)
print(np.dot(A,sol))

# x,y = np.meshgrid(np.arange(0,6),np.arange(0,6))
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(x,y, A[0])
# mul = np.dot(A_inv,np.transpose(X[0:size,0]))
#
# plt.plot(mode_mag_range,Y[0:size,0])
# plt.plot(mode_mag_range,mul[0,:])
# plt.show()
# mul = np.dot(Y,A_inv)
# print(mul)

plt.subplot(3,3,1)
fx = np.dot(X[0:10,:],A.T)
fy = np.dot(X[10:20,:],A.T)
my = np.dot(X[20:30,:],A.T)
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),Y[0:10,0:2])
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),fx[:,0:2])
plt.subplot(3,3,2)
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),Y[0:10,2:4])
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),fx[:,2:4])
plt.subplot(3,3,3)
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),Y[0:10,4:6])
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),fx[:,4:6])
plt.subplot(3,3,4)
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),Y[10:20,0:2])
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),fy[:,0:2])
plt.subplot(3,3,5)
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),Y[10:20,2:4])
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),fy[:,2:4])
plt.subplot(3,3,6)
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),Y[10:20,4:6])
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),fy[:,4:6])
plt.subplot(3,3,7)
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),Y[20:30,0:2])
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),my[:,0:2])
plt.subplot(3,3,8)
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),Y[20:30,2:4])
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),my[:,2:4])
plt.subplot(3,3,9)
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),Y[20:30,4:6])
plt.plot(np.hstack((mode_mag_range1, mode_mag_range2)),my[:,4:6])
# plt.plot(np.transpose(np.vstack((mode_mag_range1,mode_mag_range2))),np.dot(X,A[0]))
# plt.plot(mul,Y)
plt.show()
