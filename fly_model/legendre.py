import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mix(color1, color2, val):
	res = []
	for c1, c2 in zip(color1,color2):
		res.append(c1*(1-val)+c2*(val))
	return res

# Import CSV as numpy array via pandas
def npimport(file, sep=',', header=None):
	df = pd.read_csv(file, sep=sep, header=header)
	arr = np.asarray(df)
	return arr

# Kinematic time course y is calculated via y = X*(a+bx)
#	X = conversion matrix from Legendre coeffs to kinematics (mxn)
#	a = Legendre coeffs for hover flight (nx1)
#	b = Legendre coeffs for control mode (nx1)
#	x = magnitude of control mode (scalar)

black = [0,0,0]
blue = [0,0.5,0.9]
white = [1,1,1]
X = npimport('legendre/X_theta.csv')
a_theta = npimport('legendre/a_theta.csv')
b_FyL_thetaL = npimport('legendre/b_FyL_thetaL.csv')

ix = np.arange(26)
a = a_theta.flatten()
b = b_FyL_thetaL.flatten()*1e-5

# plt.bar(ix, a, 1, color=blue, edgecolor=white)
# plt.bar(ix, b, 1, bottom=a, color=mix(blue,black,0.2), edgecolor=white)
# plt.bar(ix, b, 1, bottom=a+b, color=mix(blue,black,0.4), edgecolor=white)
# plt.bar(ix, b, 1, bottom=a+2*b, color=mix(blue,black,0.6), edgecolor=white)

# plt.gca().spines["right"].set_visible(False)
# plt.gca().spines["top"].set_visible(False)
# plt.gca().spines["bottom"].set_visible(False)
# plt.gca().axes.xaxis.set_visible(False)
# plt.ylim([-0.2,0.25])
# plt.ylabel("Legendre Coefficients")
# plt.show()

plt.plot(X.dot(a_theta+b_FyL_thetaL*0), color=blue, linewidth=3)
plt.plot(X.dot(a_theta+b_FyL_thetaL*1e-5), color=mix(blue,black,0.2), linewidth=3)
plt.plot(X.dot(a_theta+b_FyL_thetaL*2e-5), color=mix(blue,black,0.4), linewidth=3)
plt.plot(X.dot(a_theta+b_FyL_thetaL*3e-5), color=mix(blue,black,0.6), linewidth=3)

plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().axes.xaxis.set_visible(False)
plt.ylim([-0.3,0.1])
plt.ylabel("Kinematics")
plt.show()