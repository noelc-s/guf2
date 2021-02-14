q0 = [0 1];
x_des = @(t) sin(t);
v_des = @(t) cos(t);
u_des = @(t) -sin(t);

tspan = [0 10]

[t,x] = ode45(@(t,x) [0 1;0 0]*x + [0; 1]*u_des(t),tspan,q0);

plot(t,x)


