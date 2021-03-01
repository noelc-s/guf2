import numpy as np

class Controller(object):
    def __init__(self,controlType,Gains):
        self.control = {
            'PD': self.PD,
            'FL': self.FL,
            'Zero': self.Zero,
        }
        self.controlType = controlType
        self.u = 0
        self.Gains = Gains

        self.q_des = np.array([0,0,4,0,0,0])
        self.dq_des = np.array([0,0,0,0,0,0])

    def getControl(self, Fly, q, dq):

        if Fly.i%Fly.wk_len == 0:
            self.u = self.control[self.controlType](Fly, q,dq)
        return self.u

    def PD(self, Fly, q, dq):

        self.q_des[0:2] = q[0:2]
        n1 = q - self.q_des
        n2 = dq - self.dq_des

        u = - self.Gains[0]*n1 - self.Gains[1]*n2

        return u

    def FL(self, Fly, q, dq):
        return [0,0,0,0,0,0]

    def Zero(self, Fly, q, dq):
        return [0,0,0,0,0,0]
    #
    #     R_eul = [1 sin(phi)*tan(theta) cos(phi)*tan(theta); 0 cos(phi) -sin(phi); 0 sin(phi)*sec(theta) cos(phi)*sec(theta)]
    #     Rz = @(t) [cos(t) sin(t) 0; -sin(t) cos(t) 0; 0 0 1]
    #     Ry = @(t) [cos(t) 0 -sin(t); 0 1 0;sin(t) 0 cos(t)]
    #     Rx = @(t) [1 0 0; 0 cos(t) sin(t); 0 -sin(t) cos(t)]
    #
    #     R = Rz(psi)*Ry(phi)*Rx(theta)
    #
    #     I = eye(3)
    #     m = 1
    #
    #     f = [x(7:9,i); R_eul*x(10:12,i);[0; 0; -g];- np.linalg.solve(I,cross(x(4:6,i),I*x(4:6,i)))]
    #     g = [zeros(6,6); 1/m*R zeros(3); zeros(3,3) np.linalg.inv(I)]
    #
    #     dy_dx = eye(12)
    #
    #     Lfh = dy_dx*f
    #     Lgh = dy_dx*g
    #
    #     n1 = q - self.q_des
    #     n2 = dq - self.dq_des
    #
    #     v = - self.Gains[1]*n1 - self.Gains[2]*n2
    #
    #     u = np.linalg.solve(Lgh,-Lfh+v)
    #
    #     return u




