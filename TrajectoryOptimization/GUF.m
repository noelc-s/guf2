
tspan = [0 10];

% q0 = [0 0 1 0 0 0]; % x y z theta phi psi
% dq0 = [0 0 0 0 0 0]; % dx dy dz dtheta dphi dpsi
% q0 = rand(1,6); % x y z theta phi psi
% dq0 = rand(1,6); % dx dy dz dtheta dphi dpsi
q0 = [ 0.1450    0.8530    0.6221    0.3510    0.5132    0.4018];
dq0 = q0;


ic = [q0 dq0]

% traj.t = [0 1];
% traj.u = [0 0;0 0;9.81 9.81];

% [t,x] = ode45(@(t,x) ODE(t,x,interp1(traj.t,traj.u',t)'),tspan, ic);
[t,x] = ode45(@(t,x) ODE(t,x,controller(t,x)),tspan, ic);



% plot3(x(:,1),x(:,2),x(:,3))
plot(t,x,'linewidth',2)
yline(0)
xlabel('$t$','interpreter','latex')
ylabel('$x$','interpreter','latex')
set(gca,'TickLabelInterpreter', 'latex');
set(gca,'FontSize',17)
set(gca,'linewidth',2)
%%
for i = 1:length(t)
    u(i,:) =  controller(t(i),x(i,:)')';
end
plot(t,u,'linewidth',2)
yline(0)
xlabel('$t$','interpreter','latex')
ylabel('$u$','interpreter','latex')
set(gca,'TickLabelInterpreter', 'latex');
set(gca,'FontSize',17)
set(gca,'linewidth',2)
legend({'$F_x$','$F_y$','$F_z$','$M_x$','$M_y$','$M_z$'},'interpreter','latex')
