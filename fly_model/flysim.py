import pybullet as p
import time
import pybullet_data
import pandas as pd
import numpy as np
import qarray

# Initialize physics and scene
physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.resetDebugVisualizerCamera(6,50,-35,[0,0,2])
# p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")

# Load fly model
flyStartPos = [0,0,3]
flyStartOrientation = p.getQuaternionFromEuler([0,0,0])
# flyId = p.loadSDF("fly.sdf")[0]
flyId = p.loadURDF("fly2.urdf", flyStartPos, flyStartOrientation)

# Generate link and joint index dictionaries
num_joints = p.getNumJoints(flyId)
link_dict = {}
joint_dict = {}
for i in range(num_joints):
    joint_info = p.getJointInfo(flyId, i)
    print(joint_info)
    link_dict[joint_info[12].decode('ascii')] = joint_info[0]
    joint_dict[joint_info[1].decode('ascii')] = joint_info[0]
print(link_dict)
print(joint_dict)

# Determine forces via quasi-steady model
def qs_force(bodyId, linkId):

    state = p.getLinkState(bodyId, linkId)

    return state

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
wk_len = wingkin.shape[0]

# Generate joint index list for arrayed motor control
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

    # p.applyExternalForce(
    #     flyId,
    #     link_dict["wingL"],
    #     [0,0,1],
    #     [0,0,0],
    #     p.LINK_FRAME
    #     )

    # if i%10==0:
    #     state = qs_force(flyId, link_dict["wingL"])
    #     for i in range(num_joints):
    #         state = p.getLinkState(flyId, i)
    #         print(state[0])
    #     # state = p.getLinkState(flyId, link_dict["wingL"])
    #     # print(state[1])

    p.stepSimulation()
    time.sleep(1./240.)

flyPos, flyOrn = p.getBasePositionAndOrientation(flyId)
print(flyPos,flyOrn)
p.disconnect()
