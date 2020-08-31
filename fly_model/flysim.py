import pybullet as p
import time
import pybullet_data
import pandas as pd
import numpy as np
import qarray

# Initialize physics and scene
physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
# p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")

# Load fly model
flyStartPos = [0,0,3]
flyStartOrientation = p.getQuaternionFromEuler([0,0,0])
# flyId = p.loadSDF("fly.sdf")[0]
flyId = p.loadURDF("fly.urdf", flyStartPos, flyStartOrientation)

# Generate joint index dictionary
num_joints = p.getNumJoints(flyId)
joint_dict = {}
for i in range(num_joints):
    joint_info = p.getJointInfo(flyId, i)
    print(joint_info)
    joint_dict[joint_info[1].decode('ascii')] = joint_info[0]
print(joint_dict)

# Load wing kinematics from file
wingkin = pd.read_csv('yan.csv', sep=' ', header=None)
wingkin = np.asarray(wingkin)
wingkin = wingkin*np.pi/180  # Convert degrees to radians

# Arrange kinematics to describe 6 DOF (3/wing)
position,rotation,deviation = wingkin.T
wingkin = np.hstack((
    position[:,None], # Left wing
    deviation[:,None],
    rotation[:,None],
    -position[:,None], # Right wing
    -deviation[:,None],
    rotation[:,None]
))
# wingkinL = np.hstack((position[:,None], deviation[:,None], rotation[:,None]))
# wingkinR = np.hstack((-position[:,None], -deviation[:,None], rotation[:,None]))

wk_len = wingkin.shape[0]

sign = 1

# p.resetJointState(flyId, joint_dict["hingeL-z"], -1)
# p.resetJointState(flyId, joint_dict["hingeR-z"], 1)

# Generate joint index list for motor control
motor_list = [
    joint_dict["hingeL-pos"],
    joint_dict["hingeL-dev"],
    joint_dict["hingeL-rot"],
    joint_dict["hingeR-pos"],
    joint_dict["hingeR-dev"],
    joint_dict["hingeR-rot"]
]

# Run simulation
for i in range (10000):

    p.setJointMotorControlArray(
        flyId,
        motor_list,
        controlMode=p.POSITION_CONTROL,
        targetPositions=wingkin[i%wk_len,:],
        forces=[500]*6
        )

    p.stepSimulation()
    time.sleep(1./240.)

flyPos, flyOrn = p.getBasePositionAndOrientation(flyId)
print(flyPos,flyOrn)
p.disconnect()
