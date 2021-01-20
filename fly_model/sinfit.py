import numpy as np
import scipy.optimize as spo
import pandas as pd
import matplotlib.pyplot as plt

lift = np.asarray(pd.read_csv('data/wing_lift.csv', header=None))
drag = np.asarray(pd.read_csv('data/wing_drag.csv', header=None))

# Define model and x-range
f = lambda x, a, b, w, c: a*np.sin(x*w) + b*np.cos(x*w) + c
x = np.arange(-10,100,1)

# Fit model for lift coefficient and plot
mdl, _ = spo.curve_fit(f, lift[:,0], lift[:,1], p0=(2,0,2*np.pi/80,0))
a,b,w,c = mdl
A = np.sqrt(a**2+b**2)
phi = np.arctan2(b,a)
print(A,w,phi,c)
plt.plot(lift[:,0], lift[:,1], 'b.', x, A*np.sin(x*w+phi)+c, 'b--')

# Fit model for drag coefficient and plot
mdl, _ = spo.curve_fit(f, drag[:,0], drag[:,1], p0=(1,1,0.0361,2))
a,b,w,c = mdl
A = np.sqrt(a**2+b**2)
phi = np.arctan2(b,a)
print(A,w,phi,c)
plt.plot(drag[:,0], drag[:,1], 'r.', x, A*np.sin(x*w+phi)+c, 'r--')

# Show results
plt.xlabel('Angle of Attack (deg)')
plt.ylabel('Force Coefficient')
plt.legend(['Lift (data)', 'Lift (model)', 'Drag (data)', 'Drag(model)'])
plt.show()