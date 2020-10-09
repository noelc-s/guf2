import pybullet as p
import time
import pybullet_data
import pandas as pd
import numpy as np
import csv
from qfunc import *
# import qarray

class Fly(object):
    
    def __init__(self, startPos, startOrn, wingkin_file):

        # Initialize physics and scene
        self.physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0) # remove overlay
        self.dt = 1./240. # seconds
        self.dt = self.dt/4
        p.setTimeStep(self.dt)

        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.resetDebugVisualizerCamera(6,50,-35,[0,0,2])
        # p.setGravity(0,0,-3)
        self.planeId = p.loadURDF("plane.urdf")

        # Load fly model
        
        # flyId = p.loadSDF("fly.sdf")[0]
        self.flyId = p.loadURDF("fly.urdf", startPos, startOrn)

        # Load markers
        self.mkRedId1 = p.loadURDF("arrow_red.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
        self.mkGrnId1 = p.loadURDF("arrow_green.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
        self.mkBluId1 = p.loadURDF("arrow_blue.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
        self.mkRedId2 = p.loadURDF("arrow_red.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
        self.mkGrnId2 = p.loadURDF("arrow_green.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
        self.mkBluId2 = p.loadURDF("arrow_blue.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))

        # Generate link and joint index dictionaries
        num_joints = p.getNumJoints(self.flyId)
        self.link_dict = {}
        self.joint_dict = {}
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.flyId, i)
            self.link_dict[joint_info[12].decode('ascii')] = joint_info[0]
            self.joint_dict[joint_info[1].decode('ascii')] = joint_info[0]

        # Load wing kinematics from file
        self.wingkin = pd.read_csv(wingkin_file, sep=' ', header=None)
        self.wingkin = np.asarray(self.wingkin)
        self.wingkin = self.wingkin*np.pi/180  # Convert degrees to radians
        # self.wingkin = self.wingkin[::2] # Subsample kinematics

        # Arrange kinematics to describe 6 DOF (3/wing)
        position,rotation,deviation = self.wingkin.T
        self.wingkin = np.hstack((
            position[:,None], # Left wing
            deviation[:,None],
            rotation[:,None],
            -position[:,None], # Right wing
            -deviation[:,None],
            rotation[:,None]
        ))
        # wingkin = wingkin*0
        self.wk_len = self.wingkin.shape[0]

        # Generate joint index list for arrayed motor control
        self.motor_list = [
            self.joint_dict["hingeL-pos"],
            self.joint_dict["hingeL-dev"],
            self.joint_dict["hingeL-rot"],
            self.joint_dict["hingeR-pos"],
            self.joint_dict["hingeR-dev"],
            self.joint_dict["hingeR-rot"]
        ]

        self.sim_time = 0

    def __del__(self):

        p.disconnect()

    def step_simulation(self):

        pitch_factor = np.array([-0.5,0,0,-.5,0,0])/10
        pitch_shift = np.array([-1,0,0,1,0,0])*0.072
        
        # target = wingkin[i%wk_len,:]
        target = self.wingkin[i%self.wk_len,:] * np.exp(-pitch_factor)
        target = target + pitch_shift

        target = self.linramp(self.sim_time, target, 1.0)

        flip = np.sign(self.wingkin[i%self.wk_len,0] - self.wingkin[i%self.wk_len-1,0])

        # Loop over wing kinematics
        p.setJointMotorControlArray(
            self.flyId,
            self.motor_list,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target,
            forces=[50000]*6
            )

        ##### Apply quasisteady model to left wing #####

        # Draw wing velocity and direction vectors
        wlVel = p.getLinkState(self.flyId, self.link_dict["wingL"], computeLinkVelocity=1)[6]
        wing_pos, vel_orn = worldPlusVector(self.flyId, self.link_dict["wingL"], [0,0,1])

        p.resetBasePositionAndOrientation(self.mkBluId1, wing_pos, vel_orn)
        p.resetBasePositionAndOrientation(self.mkRedId1, wing_pos, vec2q(wlVel))

        # Calculate angle of attack
        aoa = qAngle(
            worldPlusVector(self.flyId,self.link_dict["wingL"],[0,0,1])[1],
            vec2q(wlVel)
        )

        # Apply lift & drag forces
        span = q2vec(worldPlusVector(self.flyId, self.link_dict["wingL"], [0,1,0])[1])
        drag = self.calc_drag(aoa, wlVel)
        lift = self.calc_lift(aoa, wlVel, span, flip)
        # if not init:
        if True:
            p.applyExternalForce(
                self.flyId,
                self.link_dict["wingL"],
                lift+drag,
                p.getLinkState(self.flyId, self.link_dict["wingL"])[0],
                p.WORLD_FRAME
                )
        
        p.resetBasePositionAndOrientation(self.mkGrnId1, wing_pos, vec2q(lift+drag))

        ##### Apply quasisteady model to right wing #####

        # Draw wing velocity and direction vectors
        wlVel = p.getLinkState(self.flyId, self.link_dict["wingR"], computeLinkVelocity=1)[6]
        wing_pos, vel_orn = worldPlusVector(self.flyId, self.link_dict["wingR"], [0,0,1])

        p.resetBasePositionAndOrientation(self.mkBluId2, wing_pos, vel_orn)
        p.resetBasePositionAndOrientation(self.mkRedId2, wing_pos, vec2q(wlVel))

        # Calculate angle of attack
        aoa = qAngle(
            worldPlusVector(self.flyId,self.link_dict["wingR"],[0,0,1])[1],
            vec2q(wlVel)
        )

        # Apply lift & drag forces
        span = q2vec(worldPlusVector(self.flyId, self.link_dict["wingR"], [0,1,0])[1])
        drag = self.calc_drag(aoa, wlVel)
        lift = self.calc_lift(aoa, wlVel, span, flip)
        # if i%10==0:
        #     print(lift)
        # if not init:
        if True:
            p.applyExternalForce(
                self.flyId,
                self.link_dict["wingR"],
                lift+drag,
                p.getLinkState(self.flyId, self.link_dict["wingR"])[0],
                p.WORLD_FRAME
                )

        p.resetBasePositionAndOrientation(self.mkGrnId2, wing_pos, vec2q(lift+drag))

        p.stepSimulation()
        self.sim_time += self.dt
        time.sleep(self.dt)
    
    @staticmethod
    def calc_drag(aoa, vel):
        vel = np.array(vel)
        ang = vel / np.linalg.norm(vel)

        cD = 1.464 * np.sin(0.0342*aoa - 1.667) + 2.008
        drag = -cD * vel * np.linalg.norm(vel) * 0.1
        return drag

    @staticmethod
    def calc_lift(aoa, vel, span, flip=1):
        vel = np.array(vel)

        cL = 1.597 * np.sin(0.0407*aoa - 0.369) + 0.317
        lift = cL * np.cross(vel, flip*span) * np.linalg.norm(vel) * 0.1
        return(lift)

    @staticmethod
    def linramp(t,x,tc):
        if t < tc:
            x_ramp = x*t/tc
        else:
            x_ramp = x
        return x_ramp


if __name__ == "__main__":

    flyStartPos = [0,0,4]
    flyStartOrn = p.getQuaternionFromEuler([0,0,0])
    fly = Fly(flyStartPos, flyStartOrn, 'yan.csv')
    for i in range(10000):
        fly.step_simulation()







# Determine forces via quasi-steady model
def qs_force(bodyId, linkId):

    state = p.getLinkState(bodyId, linkId)

    return state







# data = np.empty([0,3])
# init = True
# sim_time = 0
# # Run simulation
# for i in range (0,10000,2):



    

    

    # test = q2vec(new_orn)
    # print("test: "+str(test))

    # print(wlVel)

    # state = p.getLinkState(flyId, link_dict["head"], computeLinkVelocity=1)
    # orn = state[1]
    # impulse = state[6]
    # if i%wk_len == 0:
    #     print(impulse)

#     angularVelocity = p.getBaseVelocity(flyId)[1]
#     if i%wk_len == 0:
#         print(np.array([angularVelocity]))
#         data = np.append(data, np.array([angularVelocity]), 0)


#     if init and i>50: init=False

    
#     # time.sleep(1./240.)

# print(data)
# with open('stroke_impulse.csv', mode='w') as file:
#     writer = csv.writer(file)
#     for row in data:
#         writer.writerow(row)

# flyPos, flyOrn = p.getBasePositionAndOrientation(flyId)
# print(flyPos,flyOrn)

