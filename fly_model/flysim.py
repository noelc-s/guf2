import pybullet as p
import time
import pybullet_data

# Initialize physics and scene
physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")

# Load fly model
flyStartPos = [0,0,3]
flyStartOrientation = p.getQuaternionFromEuler([.75,0,0])
flyId = p.loadURDF("fly.urdf", flyStartPos, flyStartOrientation)

nj = p.getNumJoints(flyId)
for i in range(nj):
    ji = p.getJointInfo(flyId, i)
    print(ji[:2])

sign = 1

p.resetJointState(flyId, 4, -1)
p.resetJointState(flyId, 7, 1)

# Run simulation
for i in range (10000):

    if i%100==0:
        p.setJointMotorControl2(
            flyId,4,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=5*sign,
            force=500
            )
        p.setJointMotorControl2(
            flyId,7,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-5*sign,
            force=500
            )
        sign = -sign

    js = p.getJointState(flyId,2)
    # print(js)

    p.stepSimulation()
    time.sleep(1./240.)

flyPos, flyOrn = p.getBasePositionAndOrientation(flyId)
print(flyPos,flyOrn)
p.disconnect()
