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
flyStartOrientation = p.getQuaternionFromEuler([0,0,0])
flyId = p.loadURDF("fly.urdf", flyStartPos, flyStartOrientation)

# Run simulation
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)

flyPos, flyOrn = p.getBasePositionAndOrientation(flyId)
print(flyPos,flyOrn)
p.disconnect()
