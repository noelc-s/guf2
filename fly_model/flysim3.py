import pybullet as p
import time
import pybullet_data
import pandas as pd
import numpy as np
import csv
from qfunc import *
from controller import *
import matplotlib.pyplot as plt
import modes
# import qarray

class Fly(object):

    def __init__(self, startPos, startOrn, startLinVel, startAngVel, dt, gui=True, apply_forces=True, cmd=(0,0,0,0,0,0),controller='Zero',gains=[0,0]):
        self.i = 0
        self.cmd = cmd

        self.apply_forces = apply_forces

        self.global_state = np.empty((0,6)) # x, y ,z, roll, pitch, yaw
        self.hinge_state = np.empty((0,6)) # posL, devL, rotL, posR, devR, rotR

        self.forces = np.empty((0,3)) # Fx, Fy, Fz
        self.torques = np.empty((0,3)) # Tx, Ty, Tz

        self.legendre = {}

        self.controller = Controller(controller, gains)

        # Initialize physics and scene
        if gui:
            self.physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0) # remove overlay

        self.dt = dt
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
        p.resetBaseVelocity(self.flyId, startLinVel, startAngVel)

        # Load markers
        self.mkRedId1 = p.loadURDF("Model/arrow_red.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
        self.mkGrnId1 = p.loadURDF("Model/arrow_green.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
        self.mkBluId1 = p.loadURDF("Model/arrow_blue.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
        self.mkRedId2 = p.loadURDF("Model/arrow_red.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
        self.mkGrnId2 = p.loadURDF("Model/arrow_green.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))
        self.mkBluId2 = p.loadURDF("Model/arrow_blue.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0]))

        # Generate link and joint index dictionaries
        num_joints = p.getNumJoints(self.flyId)
        self.link_dict = {}
        self.joint_dict = {}

        for i in range(num_joints):
            joint_info = p.getJointInfo(self.flyId, i)
            self.link_dict[joint_info[12].decode('ascii')] = joint_info[0]
            self.joint_dict[joint_info[1].decode('ascii')] = joint_info[0]

        self.a, self.X, self.b = modes.read_modes()
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

        self.sim_time = 0

    def step_simulation(self):
        net_force = np.zeros(3)
        net_torque = np.zeros(3)

        target, target_last = self.calc_legendre(self.i%self.wk_len) #from legendre model

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

        if self.apply_forces and self.i>0:
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
        if self.apply_forces and i>0:
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
        self.hinge_state = np.append(self.hinge_state, target.T[None,:], 0)

        self.i += 1

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

    def calc_legendre(self, i):

        kin = modes.calc_kinematics(self.a, self.X, self.b, self.cmd)

        # Adjust reference frames
        kin[:,2] -= np.pi/2
        kin[:,5] -= np.pi/2
        kin[:,3:5] *= -1

        return kin[i,:], kin[i-1,:]

    def get_control(self):
        pos_or = np.array(p.getBasePositionAndOrientation(self.flyId))
        pos = pos_or[0]
        orientation = p.getEulerFromQuaternion(pos_or[1])
        q = np.concatenate((np.array(pos),np.array(orientation)),axis=0)
        dq = np.array([0,0,0,0,0,0])
        u = self.controller.getControl(self, q, dq)
        self.cmd = u


if __name__ == "__main__":

    flyStartPos = [0,0,4]
    flyStartOrn = p.getQuaternionFromEuler([0,0,0])
    flyStartLinVel = [-1,0,0]
    flyStartAngVel = [0,0,0]

    dt = 1./300. # seconds
    tspan = 20.

    gains = [4*0.00001,0.1*0.00001]

    fly = Fly(flyStartPos, flyStartOrn,flyStartLinVel,flyStartAngVel, dt, gui=True, apply_forces=True, cmd=[0.0,0.0,0.0,0.0,0.0,0.0], controller='PD',gains = gains)
    for i in range(int(tspan/dt)):
        fly.get_control()
        fly.step_simulation()

    f = fly.forces[:,2]
    # npwrite(f,'temp/fz_00.csv')

    wb = np.arange(fly.forces.shape[0])/100

    # Plot forces and torques
    # plt.subplot(211)
    # plt.plot(wb, fly.forces, '.-')
    # plt.title("Net Forces")
    # plt.xlabel("Time (wingbeats)")
    # plt.ylabel("Force (au)")
    # plt.legend(["$F_x$", "$F_y$", "$F_z$"])
    #
    # plt.subplot(212)
    # plt.plot(wb, fly.torques, '.-')
    # plt.title("Net Torques")
    # plt.xlabel("Time (wingbeats)")
    # plt.ylabel("Torque (au)")
    # plt.legend(["Roll ($\\tau_x$)", "Pitch ($\\tau_y$)", "Yaw ($\\tau_z$)"])
    #
    # plt.tight_layout()
    # plt.show()