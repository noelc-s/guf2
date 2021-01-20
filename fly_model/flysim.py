import pybullet as p
import time
import pybullet_data
import pandas as pd
import numpy as np
import csv
from qfunc import *
import matplotlib.pyplot as plt
# import qarray

class Fly(object):
    
    def __init__(self, startPos, startOrn, wingkin_file, apply_forces=True):

        self.i = 0

        self.apply_forces = apply_forces

        self.global_state = np.empty((0,6)) # x, y ,z, roll, pitch, yaw
        self.hinge_state = np.empty((0,6)) # posL, devL, rotL, posR, devR, rotR

        self.forces = np.empty((0,3)) # Fx, Fy, Fz
        self.torques = np.empty((0,3)) # Tx, Ty, Tz

        self.legendre = {}
        self.init_legendre()

        # Initialize physics and scene
        self.physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0) # remove overlay
        self.dt = 1./120. # seconds
        # self.dt = self.dt/4
        p.setTimeStep(self.dt)

        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.resetDebugVisualizerCamera(6,50,-35,[0,0,2])
        if self.apply_forces:
            p.setGravity(0,0,-0.5)
        else:
            p.setGravity(0,0,0)
        self.planeId = p.loadURDF("plane.urdf")

        # Load fly model
        
        # flyId = p.loadSDF("fly.sdf")[0]
        self.flyId = p.loadURDF("Model/fly.urdf", startPos, startOrn)

        # Load markers
        self.mkRedId1 = p.loadURDF("Model/arrow_red.urdf", [1, 0, 3], p.getQuaternionFromEuler([0, 0, 0]))
        self.mkGrnId1 = p.loadURDF("Model/arrow_green.urdf", [1, 0, 3], p.getQuaternionFromEuler([0, 0, 0]))
        self.mkBluId1 = p.loadURDF("Model/arrow_blue.urdf", [1, 0, 3], p.getQuaternionFromEuler([0, 0, 0]))
        self.mkRedId2 = p.loadURDF("Model/arrow_red.urdf", [1, 0, 3], p.getQuaternionFromEuler([0, 0, 0]))
        self.mkGrnId2 = p.loadURDF("Model/arrow_green.urdf", [1, 0, 3], p.getQuaternionFromEuler([0, 0, 0]))
        self.mkBluId2 = p.loadURDF("Model/arrow_blue.urdf", [1, 0, 3], p.getQuaternionFromEuler([0, 0, 0]))

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
        self.wingkin = self.wingkin[::2] # Subsample kinematics

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
        self.wk_len = 100 #from legendre kinematics

        # Generate joint index list for arrayed motor control
        self.motor_list = [
            self.joint_dict["hingeL-pos"],
            self.joint_dict["hingeL-dev"],
            self.joint_dict["hingeL-rot"],
            self.joint_dict["hingeR-pos"],
            self.joint_dict["hingeR-dev"],
            self.joint_dict["hingeR-rot"]
        ]

        for i, joint in enumerate(self.motor_list):
            p.resetJointState(self.flyId, joint, self.calc_legendre(i%self.wk_len)[0][i])

        self.calc_com()

        self.sim_time = 0

    def step_simulation(self):

        mag=0

        net_force = np.zeros(3)
        net_torque = np.zeros(3)

        pitch_factor = np.array([-0.5,0,0,-.5,0,0])/10
        pitch_shift = np.array([-1,0,0,1,0,0])*0.072
        
        # target = wingkin[i%wk_len,:]
        # target = self.wingkin[i%self.wk_len,:] * np.exp(-pitch_factor)
        # target = target + pitch_shift
        target, target_last = self.calc_legendre(self.i%self.wk_len, mag) #from legendre model

        # Chirp kinematics if forces are being applied
        if self.apply_forces:
            target = self.linramp(self.sim_time, target, 1.0)

        flip = np.sign(target[0] - target_last[0])

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
        
        if self.apply_forces:
            p.applyExternalForce(
                self.flyId,
                self.link_dict["wingL"],
                lift+drag,
                p.getLinkState(self.flyId, self.link_dict["wingL"])[0],
                p.WORLD_FRAME
                )

        # Accumulate forces and torques
        net_force += lift+drag
        lever = np.array(p.getLinkState(self.flyId, self.link_dict["wingL"])[0]) - np.array(p.getBasePositionAndOrientation(self.flyId)[0])
        net_torque += np.cross(lever, lift+drag)
        
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
        if self.apply_forces:
            p.applyExternalForce(
                self.flyId,
                self.link_dict["wingR"],
                lift+drag,
                p.getLinkState(self.flyId, self.link_dict["wingR"])[0],
                p.WORLD_FRAME
                )

        net_force += lift+drag
        
        lever = np.array(p.getLinkState(self.flyId, self.link_dict["wingR"])[0]) - np.array(p.getBasePositionAndOrientation(self.flyId)[0])
        net_torque += np.cross(lever, lift+drag)

        p.resetBasePositionAndOrientation(self.mkGrnId2, wing_pos, vec2q(lift+drag))

        p.stepSimulation()
        self.sim_time += self.dt
        time.sleep(self.dt)

        self.forces = np.append(self.forces, net_force[None,:], 0)
        self.torques = np.append(self.torques, net_torque[None,:], 0)
        self.hinge_state = np.append(self.hinge_state, target.T, 0)

        self.i += 1
    
    def calc_com(self):
        
        for link in self.link_dict:
            # print(link)
            pass

    @staticmethod
    def calc_drag(aoa, vel):
        vel = np.array(vel)
        mag = np.linalg.norm(vel)
        if mag != 0:
            ang = vel / mag
        else:
            ang = vel

        cD = 1.464 * np.sin(0.0342*aoa - 1.667) + 2.008
        drag = -cD * vel * np.linalg.norm(vel) * 0.5
        return drag

    @staticmethod
    def calc_lift(aoa, vel, span, flip=1):
        vel = np.array(vel)

        cL = 1.597 * np.sin(0.0407*aoa - 0.369) + 0.317
        lift = cL * np.cross(vel, flip*span) * np.linalg.norm(vel) * 0.5
        return(lift)

    @staticmethod
    def linramp(t,x,tc):
        if t < tc:
            x_ramp = x*t/tc
        else:
            x_ramp = x
        return x_ramp

    # Load Legendre wing kinematics model
    def init_legendre(self):

        print("*******************************")
        print("Running legendre initialization")
        print("*******************************")

        # Import model coefficients
        conversion_matrices = [
            "X_theta",
            "X_eta",
            "X_phi"
        ]
        for matrix in conversion_matrices:
            self.legendre[matrix] = npread('legendre/hover/{}.csv'.format(matrix))

        hover_matrices = [
            "a_theta",
            "a_eta",
            "a_phi"
        ]
        for matrix in hover_matrices:
            self.legendre[matrix] = npread('legendre/hover/{}.csv'.format(matrix))

        control_matrices = [
            "b_FzU_theta",
            "b_FzU_eta",
            "b_FzU_phi"
        ]
        for matrix in control_matrices:
            self.legendre[matrix] = npread('legendre/forces/{}.csv'.format(matrix))

    def calc_legendre(self, i, mag=0):

        # Kinematic time course y is calculated via y = X*(a+bx)
            #	X = conversion matrix from Legendre coeffs to kinematics (mxn)
            #	a = Legendre coeffs for hover flight (nx1)
            #	b = Legendre coeffs for control mode (nx1)
            #	x = magnitude of control mode (scalar)
        
        posL = self.legendre["X_phi"].dot(self.legendre["a_phi"]+mag*self.legendre["b_FzU_phi"])
        devL = self.legendre["X_theta"].dot(self.legendre["a_theta"]+mag*self.legendre["b_FzU_theta"])
        rotL = self.legendre["X_eta"].dot(self.legendre["a_eta"]+mag*self.legendre["b_FzU_eta"])-np.pi/2
        posR = -self.legendre["X_phi"].dot(self.legendre["a_phi"]+mag*self.legendre["b_FzU_phi"])
        devR = -self.legendre["X_theta"].dot(self.legendre["a_theta"]+mag*self.legendre["b_FzU_theta"])
        rotR = self.legendre["X_eta"].dot(self.legendre["a_eta"]+mag*self.legendre["b_FzU_eta"])-np.pi/2

        # wingkin = np.hstack((
        #     self.legendre["X_phi"].dot(self.legendre["a_phi"]),
        #     self.legendre["X_theta"].dot(self.legendre["a_theta"]),
        #     self.legendre["X_eta"].dot(self.legendre["a_eta"])-np.pi/2,
        #     -self.legendre["X_phi"].dot(self.legendre["a_phi"]),
        #     -self.legendre["X_theta"].dot(self.legendre["a_theta"]),
        #     self.legendre["X_eta"].dot(self.legendre["a_eta"])-np.pi/2
        # ))

        wk = np.array([posL[i], devL[i], rotL[i], posR[i], devR[i], rotR[i]])
        wk_last = np.array([posL[i-1], devL[i-1], rotL[i-1], posR[i-1], devR[i-1], rotR[i-1]])

        return wk, wk_last

    def __del__(self):

        p.disconnect()

# Import CSV as numpy array via pandas
def npread(file, sep=',', header=None):
    df = pd.read_csv(file, sep=sep, header=header)
    arr = np.asarray(df)
    return arr

def npwrite(arr, file):
    df = pd.DataFrame(arr)
    df.to_csv(file, header=False, index=False)

if __name__ == "__main__":

    flyStartPos = [0,0,4]
    flyStartOrn = p.getQuaternionFromEuler([0,0,0])
    fly = Fly(flyStartPos, flyStartOrn, 'data/yan.csv', apply_forces=True)
    for i in range(1000):
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

    # plt.plot(fly.forces[:,2])
    # npwrite(fly.forces[:,2], "data/fz_ctrlp4.csv")
    # plt.show()

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

