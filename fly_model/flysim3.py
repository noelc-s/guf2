import pybullet as p
import time
import pybullet_data
import pandas as pd
import numpy as np
import csv
from qfunc import *
from controller import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import modes
# import qarray

class Fly(object):

    def __init__(self, startPos, startOrn, startLinVel, startAngVel, dt, gui=True, apply_forces=True, cmd=(0,0,0,0,0,0),controller='Zero',gains=[0,0]):

        self.apply_forces = apply_forces

        # Initialize physics and scene
        if gui:
            self.physicsClient = p.connect(p.GUI) # with GUI
        else:
            self.physicsClient = p.connect(p.DIRECT) # without gui
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0) # remove overlay

        # Integration time
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

        self.flyId = p.loadURDF("Model/fly.urdf", startPos, startOrn)

        self.mkRedId = []
        self.mkGrnId = []
        self.mkBluId = []

        # Load markers (for forces)
        self.mkRedId.append(p.loadURDF("Model/arrow_red.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0])))
        self.mkGrnId.append(p.loadURDF("Model/arrow_green.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0])))
        self.mkBluId.append(p.loadURDF("Model/arrow_blue.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0])))
        self.mkRedId.append(p.loadURDF("Model/arrow_red.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0])))
        self.mkGrnId.append(p.loadURDF("Model/arrow_green.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0])))
        self.mkBluId.append(p.loadURDF("Model/arrow_blue.urdf", [1,0,3], p.getQuaternionFromEuler([0,0,0])))

        # Generate link and joint index dictionaries
        num_joints = p.getNumJoints(self.flyId)
        self.link_dict = {}
        self.joint_dict = {}

        for i in range(num_joints):
            joint_info = p.getJointInfo(self.flyId, i)
            self.link_dict[joint_info[12].decode('ascii')] = joint_info[0]
            self.joint_dict[joint_info[1].decode('ascii')] = joint_info[0]

        # Initialize legendre modes
        self.a, self.X, self.b = modes.read_modes()
        self.wk_len = 100 # number of data points per wingbeat, from legendre kinematics

        # Generate joint index list for arrayed motor control
        self.motor_list = [
            self.joint_dict["hingeL-pos"],
            self.joint_dict["hingeL-dev"],
            self.joint_dict["hingeL-rot"],
            self.joint_dict["hingeR-pos"],
            self.joint_dict["hingeR-dev"],
            self.joint_dict["hingeR-rot"]
        ]

        # Initialize state and controller
        self.initialize(startPos, startOrn, startLinVel, startAngVel, gui, apply_forces, cmd,controller,gains)

    def initialize(self, startPos, startOrn, startLinVel, startAngVel, gui=True, apply_forces=True, cmd=(0,0,0,0,0,0),controller='Zero',gains=[0,0]):
        self.i = 0
        self.cmd = cmd
        self.sim_time = 0

        self.global_state = np.empty((0,6)) # x, y ,z, roll, pitch, yaw
        self.hinge_state = np.empty((0,6)) # posL, devL, rotL, posR, devR, rotR

        self.forces = np.empty((0,3)) # Fx, Fy, Fz
        self.torques = np.empty((0,3)) # Tx, Ty, Tz

        self.legendre = {}

        self.controller = Controller(controller, gains,cmd)

        self.target_last = np.array([0,0,0,0,0,0])

        for i, joint in enumerate(self.motor_list):
            p.resetJointState(self.flyId, joint, self.calc_legendre(0)[i])

        p.resetBasePositionAndOrientation(self.flyId, startPos,startOrn)
        p.resetBaseVelocity(self.flyId, startLinVel, startAngVel)

    def step_simulation(self):
        net_force = np.zeros(3)
        net_torque = np.zeros(3)

        target = self.calc_legendre(self.i%self.wk_len) #from legendre model

        # If ramping is desired -- causes strange behavior
        # if self.apply_forces:
        #     target = self.linramp(self.sim_time, target, 1.0)

        # keeping track of wing flip
        flip = np.sign(target[0] - self.target_last[0])

        # Loop over wing kinematics
        p.setJointMotorControlArray(
            self.flyId,
            self.motor_list,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target,
            forces=[50000]*6
        )

        ##### Apply quasisteady model to left and right wings #####
        for i, wing in enumerate(["wingL","wingR"]):
            # Draw wing velocity and direction vectors
            wlVel = p.getLinkState(self.flyId, self.link_dict[wing], computeLinkVelocity=1)[6]
            wing_pos, vel_orn = worldPlusVector(self.flyId, self.link_dict[wing], [0,0,1])

            p.resetBasePositionAndOrientation(self.mkBluId[i], wing_pos, vel_orn)
            p.resetBasePositionAndOrientation(self.mkRedId[i], wing_pos, vec2q(wlVel))

            # Calculate angle of attack
            aoa = qAngle(
                worldPlusVector(self.flyId,self.link_dict[wing],[0,0,1])[1],
                vec2q(wlVel)
            )

            # Apply lift & drag forces
            span = q2vec(worldPlusVector(self.flyId, self.link_dict[wing], [0,1,0])[1])
            drag = self.calc_drag(aoa, wlVel)
            lift = self.calc_lift(aoa, wlVel, span, flip)

            if self.apply_forces and self.i>100:
                p.applyExternalForce(
                    self.flyId,
                    self.link_dict[wing],
                    lift+drag,
                    p.getLinkState(self.flyId, self.link_dict[wing])[0],
                    p.WORLD_FRAME
                )

            # Accumulate forces and torques
            net_force += lift+drag
            lever = np.array(p.getLinkState(self.flyId, self.link_dict[wing])[0]) - np.array(p.getBasePositionAndOrientation(self.flyId)[0])
            net_torque += np.cross(lever, lift+drag)

            p.resetBasePositionAndOrientation(self.mkGrnId[i], wing_pos, vec2q(lift+drag))

        p.stepSimulation()
        self.sim_time += self.dt
        time.sleep(self.dt)

        net_force += (-1,0,-1.5)
        net_torque += (0,0.765,0)

        # TODO: This is a force offset hack
        if self.i > 100:
            p.applyExternalForce(
                self.flyId,
                -1,
                np.array((-1,0,-1.8)),
                np.array((0,0,0)),
                p.LINK_FRAME
                )
            p.applyExternalTorque(
                self.flyId,
                -1,
                np.array((0,0.786,0)),
                p.LINK_FRAME
            )

        self.forces = np.append(self.forces, net_force[None,:], 0)
        self.torques = np.append(self.torques, net_torque[None,:], 0)
        self.hinge_state = np.append(self.hinge_state, target.T[None,:], 0)

        self.i += 1

        self.target_last = target



        # Revolute only
        # p.createConstraint(self.flyId, -1, -1, -1, p.JOINT_POINT2POINT, [0,0,0], [0, 0, 0], [0,0,4])
        # p.createConstraint(self.flyId, -1, -1, -1, p.JOINT_PRISMATIC, [0,0,1], [0, 0, 0], [0,0,4])

        # pos_or = np.array(p.getBasePositionAndOrientation(self.flyId))
        # pos = pos_or[0]

        # p.resetBasePositionAndOrientation(self.flyId, pos, [0,0,0,1])

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
        return drag/100

    @staticmethod
    def calc_lift(aoa, vel, span, flip=1):
        vel = np.array(vel)

        cL = 1.597 * np.sin(0.0407*aoa - 0.369) + 0.317
        lift = cL * np.cross(vel, flip*span) * np.linalg.norm(vel) * 0.5
        return lift/100

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

        t = np.array(range(0, 100))
        f = interp1d(t, kin, axis=0)

        return f(i)

        # return kin[i,:], kin[i-1,:]

    def get_control(self):
        r2d2Vel = p.getBaseVelocity(self.flyId)
        pos_or = np.array(p.getBasePositionAndOrientation(self.flyId))
        pos = pos_or[0]
        orientation = p.getEulerFromQuaternion(pos_or[1])
        q = np.concatenate((np.array(pos),np.array(orientation)),axis=0)
        dq = np.concatenate((np.array(r2d2Vel[0]),np.array(r2d2Vel[1])),axis=0)
        u = self.controller.getControl(self, q, dq)
        self.cmd = u

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
    flyStartLinVel = [0,0,0]
    flyStartAngVel = [0,0,0]

    dt = 1./1000. # seconds
    tspan = 200/10

    # gains = np.array([10,50,1,20])*0.000001
    KP = np.array([10,10,30,50,50,50])*0.000001
    KD = np.array([1,1,10,20,20,20])*0.000001
    gains = np.concatenate((KP,KD))
    # fly = Fly(flyStartPos, flyStartOrn,flyStartLinVel,flyStartAngVel, dt, gui=True, apply_forces=True, cmd=[0.0,0.0,0.0,0.0,0.0,0.0], controller='PD',gains = gains)
    fly = Fly(flyStartPos, flyStartOrn,flyStartLinVel,flyStartAngVel, dt, gui=False, apply_forces=True, cmd=[0.0,0.0,0.0,0.0,0.0,0.0], controller='Constant',gains = gains)

    ##### Nomral integration
    # for i in range(int(tspan/dt)):
    #     fly.get_control()
    #     fly.step_simulation()
    # #
    # wb = np.arange(fly.forces.shape[0])/100
    # f = np.concatenate((fly.forces, fly.torques), axis = 1)
    # stroke_avg = []
    # for i in range(0,6):
    #     stroke_avg.append(np.mean(f[100:,i]))
    # plt.bar(range(0,6),stroke_avg)
    # plt.plot(wb[100:],f[100:,2])

    ### Automated construction of Fx,...,Mz
    directions = ['fx','fy','fz','mx','my','mz']
    for direction in range(0,6):
        for j in range(-5,5):
            cmd = [0,0,0,0,0,0]
            cmd[direction] = 0.00001*j
            fly.initialize(flyStartPos, flyStartOrn,flyStartLinVel,flyStartAngVel, gui=False, apply_forces=False, cmd=cmd, controller='Constant',gains = gains)

            for i in range(int(tspan/dt)):
                fly.get_control()
                fly.step_simulation()

            f1 = np.concatenate((fly.forces, fly.torques), axis = 1)
            npwrite(f1,'ForceCharacterization/'+directions[direction]+'_'+str(j)+'.csv')

    ## Crosstalk between modes
    # directions = ['fx','fy','fz','mx','my','mz']
    # direction1 = 0
    # name1 = directions[direction1]
    # direction2 = 2
    # name2 = directions[direction2]
    # for direction in range(0,6):
    #     for j1 in range(-5,5):
    #         for j2 in range(-5,5):
    #             cmd = [0,0,0,0,0,0]
    #             cmd[direction1] = 0.00001*j1
    #             cmd[direction2] = 0.00001*j2
    #             fly.initialize(flyStartPos, flyStartOrn,flyStartLinVel,flyStartAngVel, gui=False, apply_forces=False, cmd=cmd, controller='Constant',gains = gains)
    #
    #             for i in range(int(tspan/dt)):
    #                 fly.get_control()
    #                 fly.step_simulation()
    #
    #             f1 = np.concatenate((fly.forces, fly.torques), axis = 1)
    #             npwrite(f1,'ForceCharacterization/'+name1+'_'+name2+'_'+str(j1)+'_'+str(j2)+'.csv')

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

    plt.tight_layout()
    plt.show()