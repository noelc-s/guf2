
tspan = [0 1];

q0 = [0 0 5 0 0 0]; % x y z theta phi psi
dq0 = [0 0 0 0 0 0]; % dx dy dz dtheta dphi dpsi

ic = [q0 dq0]

% traj.t = [0 1];
% traj.u = [0 0;0 0;9.81 9.81];

[t,x] = ode45(@(t,x) ODE(t,x,interp1(traj.t,traj.u',t)'),tspan, ic);

plot3(x(:,1),x(:,2),x(:,3))