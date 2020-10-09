import pybullet as p
import time
import pybullet_data
import pandas as pd
import numpy as np
import csv
from qfunc import *
# import qarray

class Fly(object):
    pass

def calc_drag(aoa, vel):
    vel = np.array(vel)
    ang = vel / np.linalg.norm(vel)

    cD = 1.464 * np.sin(0.0342*aoa - 1.667) + 2.008
    drag = -cD * vel * np.linalg.norm(vel) * 0.1
    return drag

def calc_lift(aoa, vel, span, flip=1):
    vel = np.array(vel)

    cL = 1.597 * np.sin(0.0407*aoa - 0.369) + 0.317
    lift = cL * np.cross(vel, flip*span) * np.linalg.norm(vel) * 0.1
    return(lift)

def linramp(t,x,tc):
    if t < tc:
        x_ramp = x*t/tc
    else:
        x_ramp = x
    return x_ramp

# Initialize physics and scene
physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0) # remove overlay
dt = 1./240. # seconds
dt = dt/4
# dt = 1/(500*250)
p.setTimeStep(dt)

p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.resetDebugVisualizerCamera(6,50,-35,[0,0,2])
p.setGravity(0,0,-3)
planeId = p.loadURDF("plane.urdf")

# Load fly model
flyStartPos = [0,0,4]
flyStartOrientation = p.getQuaternionFromEuler([0,0,0])
# flyId = p.loadSDF("fly.sdf")[0]
flyId = p.loadURDF("fly.urdf", flyStartPos, flyStartOrientation)

# Load markers
mkRedId1 = p.loadURDF("arrow_red.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
mkGrnId1 = p.loadURDF("arrow_green.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
mkBluId1 = p.loadURDF("arrow_blue.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
mkRedId2 = p.loadURDF("arrow_red.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
mkGrnId2 = p.loadURDF("arrow_green.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
mkBluId2 = p.loadURDF("arrow_blue.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))

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
# wingkin = wingkin[::2] # Subsample kinematics

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
# wingkin = wingkin*0
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

data = np.empty([0,3])
init = True
sim_time = 0
# Run simulation
for i in range (0,10000,2):

    pitch_factor = np.array([-0.5,0,0,-.5,0,0])/10
    pitch_shift = np.array([-1,0,0,1,0,0])*0.072
    
    # target = wingkin[i%wk_len,:]
    target = wingkin[i%wk_len,:] * np.exp(-pitch_factor)
    target = target + pitch_shift

    target = linramp(sim_time, target, 1.0)

    flip = np.sign(wingkin[i%wk_len,0] - wingkin[i%wk_len-1,0])

    # Loop over wing kinematics
    p.setJointMotorControlArray(
        flyId,
        motor_list,
        controlMode=p.POSITION_CONTROL,
        targetPositions=target,
        forces=[50000]*6
        )

    ##### Apply quasisteady model to left wing #####

    # Draw wing velocity and direction vectors
    wlVel = p.getLinkState(flyId, link_dict["wingL"], computeLinkVelocity=1)[6]
    wing_pos, vel_orn = worldPlusVector(flyId, link_dict["wingL"], [0,0,1])

    p.resetBasePositionAndOrientation(mkBluId1, wing_pos, vel_orn)
    p.resetBasePositionAndOrientation(mkRedId1, wing_pos, vec2q(wlVel))

    # Calculate angle of attack
    aoa = qAngle(
        worldPlusVector(flyId,link_dict["wingL"],[0,0,1])[1],
        vec2q(wlVel)
    )

    # Apply lift & drag forces
    span = q2vec(worldPlusVector(flyId, link_dict["wingL"], [0,1,0])[1])
    drag = calc_drag(aoa, wlVel)
    lift = calc_lift(aoa, wlVel, span, flip)
    # if not init:
    if True:
        p.applyExternalForce(
            flyId,
            link_dict["wingL"],
            lift+drag,
            p.getLinkState(flyId, link_dict["wingL"])[0],
            p.WORLD_FRAME
            )
    
    p.resetBasePositionAndOrientation(mkGrnId1, wing_pos, vec2q(lift+drag))

    ##### Apply quasisteady model to right wing #####

    # Draw wing velocity and direction vectors
    wlVel = p.getLinkState(flyId, link_dict["wingR"], computeLinkVelocity=1)[6]
    wing_pos, vel_orn = worldPlusVector(flyId, link_dict["wingR"], [0,0,1])

    p.resetBasePositionAndOrientation(mkBluId2, wing_pos, vel_orn)
    p.resetBasePositionAndOrientation(mkRedId2, wing_pos, vec2q(wlVel))

    # Calculate angle of attack
    aoa = qAngle(
        worldPlusVector(flyId,link_dict["wingR"],[0,0,1])[1],
        vec2q(wlVel)
    )

    # Apply lift & drag forces
    span = q2vec(worldPlusVector(flyId, link_dict["wingR"], [0,1,0])[1])
    drag = calc_drag(aoa, wlVel)
    lift = calc_lift(aoa, wlVel, span, flip)
    # if i%10==0:
    #     print(lift)
    # if not init:
    if True:
        p.applyExternalForce(
            flyId,
            link_dict["wingR"],
            lift+drag,
            p.getLinkState(flyId, link_dict["wingR"])[0],
            p.WORLD_FRAME
            )

    p.resetBasePositionAndOrientation(mkGrnId2, wing_pos, vec2q(lift+drag))

    # test = q2vec(new_orn)
    # print("test: "+str(test))

    # print(wlVel)

    # state = p.getLinkState(flyId, link_dict["head"], computeLinkVelocity=1)
    # orn = state[1]
    # impulse = state[6]
    # if i%wk_len == 0:
    #     print(impulse)

    angularVelocity = p.getBaseVelocity(flyId)[1]
    if i%wk_len == 0:
        print(np.array([angularVelocity]))
        data = np.append(data, np.array([angularVelocity]), 0)


    if init and i>50: init=False

    p.stepSimulation()
    time.sleep(dt)
    sim_time += dt
    # time.sleep(1./240.)

print(data)
with open('stroke_impulse.csv', mode='w') as file:
    writer = csv.writer(file)
    for row in data:
        writer.writerow(row)

flyPos, flyOrn = p.getBasePositionAndOrientation(flyId)
print(flyPos,flyOrn)
p.disconnect()
