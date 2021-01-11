import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh
import scipy.interpolate as spi

def sliding_average(x, N):
	return np.convolve(x, np.ones((N,))/N, mode='same')

def get_inliers(arr, thresh):

	inliers = np.arange(len(arr))

	while True:

		err = np.abs(arr[inliers]-sliding_average(arr[inliers],3))
		val = np.max(err)
		ix = np.argmax(err)

		if val > thresh:
			inliers = np.concatenate((inliers[:ix], inliers[ix+1:]))
		else:
			break

	return inliers


mesh0 = mesh.Mesh.from_file("membrane_full_L.stl")

# Get unique vertices
points = np.unique(np.reshape(mesh0.points, (mesh0.points.shape[0]*3, -1)), axis=0)
points = points[:,:2]
points = points[:,::-1]

# Get edge points
top = []
bottom = []
for x in np.unique(points[:,0]):
	indices = np.where(points[:,0] == x)
	y_values = points[indices,1]
	top.append([x, np.max(y_values)])
	bottom.append([x, np.min(y_values)])
top = np.array(top)
bottom = np.array(bottom)

# Remove outliers
xt = top[:,0]
yt = top[:,1]
ixt = get_inliers(yt, 0.05)
xb = bottom[:,0]
yb = bottom[:,1]
ixb = get_inliers(yb, 0.05)

x_smooth = np.arange(0,np.max(xt),0.0025)
f_top = spi.interp1d(xt[ixt],yt[ixt],kind="cubic")
f_bottom = spi.interp1d(xb[ixb],yb[ixb],kind="cubic")

# Plot results
plt.plot(x_smooth,f_top(x_smooth))
plt.plot(x_smooth,f_bottom(x_smooth))

plt.gca().set_aspect('equal')

plt.show()
