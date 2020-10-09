import pybullet as p
import numpy as np

def q2vec(q):
    roll, pitch, yaw = p.getEulerFromQuaternion(q)
    x = np.cos(yaw)*np.cos(pitch)
    y = np.sin(yaw)*np.cos(pitch)
    z = np.sin(pitch)
    return np.array((x,y,z))

def vec2q(vec):
    if np.linalg.norm(vec)>0:
        vec = np.array(vec)/np.linalg.norm(vec)
    x, y, z = vec
    pitch = -np.arcsin(z)
    if x!=0:
        yaw = np.arctan2(y,x)
    else:
        yaw = np.pi/2 * np.sign(y)
    roll = 0
    return p.getQuaternionFromEuler((roll, pitch, yaw))

def worldPlusQuaternion(bodyId, linkId, q):
    state = p.getLinkState(bodyId, linkId)
    pos, orn = p.multiplyTransforms(
        state[0],
        state[1],
        [0,0,0],
        q
    )
    return pos, orn

def worldPlusVector(bodyId, linkId, vector):
    state = p.getLinkState(bodyId, linkId)
    pos, orn = p.multiplyTransforms(
        state[0],
        state[1],
        [0,0,0],
        vec2q(vector)
    )
    return pos, orn

def worldPlusEuler(bodyId, linkId, euler):
    state = p.getLinkState(bodyId, linkId)
    pos, orn = p.multiplyTransforms(
        state[0],
        state[1],
        [0,0,0],
        p.getQuaternionFromEuler(euler)
    )
    return pos, orn

def qAngle(q1, q2):
    v1 = q2vec(q1)
    v2 = q2vec(q2)
    return np.arccos(v1.dot(v2))