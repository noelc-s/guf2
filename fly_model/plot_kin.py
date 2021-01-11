import flysim
import pybullet as p
import matplotlib.pyplot as plt
import numpy as np

flyStartPos = [0,0,4]
flyStartOrn = p.getQuaternionFromEuler([0,0,0])
fly = flysim.Fly(flyStartPos, flyStartOrn, 'yan.csv', apply_forces=False)
for i in range(200):
	fly.step_simulation()

wb = np.arange(fly.forces.shape[0])/100

# Plot wing kinematics
plt.subplot(211)
plt.plot(wb,fly.hinge_state[:,:3], '.-')
plt.title("Left Wing Kinematics")
plt.xlabel("Time (wingbeats)")
plt.ylabel("Angle (rad)")
plt.legend(["Stroke Position", "Deviation", "Rotation"])

plt.subplot(212)
plt.plot(wb,fly.hinge_state[:,3:], '.-')
plt.title("Right Wing Kinematics")
plt.xlabel("Time (wingbeats)")
plt.ylabel("Angle (rad)")
plt.legend(["Stroke Position", "Deviation", "Rotation"])

plt.tight_layout()
plt.show()

# Plot forces and torques
plt.subplot(211)
plt.plot(wb, fly.forces, '.-')
plt.title("Net Forces")
plt.xlabel("Time (wingbeats)")
plt.ylabel("Force (au)")
plt.legend(["$F_x$", "$F_y$", "$F_z$"])

plt.subplot(212)
plt.plot(wb, fly.torques, '.-')
plt.title("Net Torques")
plt.xlabel("Time (wingbeats)")
plt.ylabel("Torque (au)")
plt.legend(["Roll ($\\tau_x$)", "Pitch ($\\tau_y$)", "Yaw ($\\tau_z$)"])

plt.tight_layout()
plt.show()