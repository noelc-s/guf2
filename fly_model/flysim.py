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

position,rotation,deviation = wingkin.T
wingkinL = np.hstack((position[:,None], -deviation[:,None], -rotation[:,None]))
wingkinR = np.hstack((-position[:,None], deviation[:,None], -rotation[:,None]))

wk_len = wingkin.shape[0]

sign = 1

# p.resetJointState(flyId, joint_dict["hingeL-z"], -1)
# p.resetJointState(flyId, joint_dict["hingeR-z"], 1)

# Run simulation
for i in range (10000):

    # # Placeholder motion
    # if i%100==0:
    #     p.setJointMotorControl2(
    #         flyId,joint_dict["hingeL-z"],
    #         controlMode=p.VELOCITY_CONTROL,
    #         targetVelocity=5*sign,
    #         force=500
    #         )
    #     p.setJointMotorControl2(
    #         flyId,joint_dict["hingeR-z"],
    #         controlMode=p.VELOCITY_CONTROL,
    #         targetVelocity=-5*sign,
    #         force=500
    #         )
    #     sign = -sign

    p.setJointMotorControl2(
        flyId,joint_dict["hingeL-pos"],
        controlMode=p.POSITION_CONTROL,
        targetPosition=wingkinL[i%wk_len,0],
        force=500
        )
    p.setJointMotorControl2(
        flyId,joint_dict["hingeL-dev"],
        controlMode=p.POSITION_CONTROL,
        targetPosition=-wingkinL[i%wk_len,1],
        force=500
        )
    p.setJointMotorControl2(
        flyId,joint_dict["hingeL-rot"],
        controlMode=p.POSITION_CONTROL,
        targetPosition=-wingkinL[i%wk_len,2],
        force=500
        )
    
    p.setJointMotorControl2(
        flyId,joint_dict["hingeR-pos"],
        controlMode=p.POSITION_CONTROL,
        targetPosition=wingkinR[i%wk_len,0],
        force=500
        )
    p.setJointMotorControl2(
        flyId,joint_dict["hingeR-dev"],
        controlMode=p.POSITION_CONTROL,
        targetPosition=-wingkinR[i%wk_len,1],
        force=500
        )
    p.setJointMotorControl2(
        flyId,joint_dict["hingeR-rot"],
        controlMode=p.POSITION_CONTROL,
        targetPosition=-wingkinR[i%wk_len,2],
        force=500
        )

    # js = p.getJointState(flyId,2)
    # print(js)

    p.stepSimulation()
    time.sleep(1./240.)

flyPos, flyOrn = p.getBasePositionAndOrientation(flyId)
print(flyPos,flyOrn)
p.disconnect()
