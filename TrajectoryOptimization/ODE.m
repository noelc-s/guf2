function dx = ODE(t,x,u)

g = 9.81;
dx = x;

for i = 1:size(x,2)

t1 = x(4,i);
t2 = x(5,i);
t3 = x(6,i);

Rz = @(t) [cos(t) sin(t) 0; -sin(t) cos(t) 0; 0 0 1];
Ry = @(t) [cos(t) 0 -sin(t); 0 1 0;sin(t) 0 cos(t)];
Rx = @(t) [1 0 0; 0 cos(t) sin(t); 0 -sin(t) cos(t)];

R = Rz(t1)*Ry(t2)*Rx(t3);

dx(1,i) = x(7,i);
dx(2,i) = x(8,i);
dx(3,i) = x(9,i);
dx(4,i) = x(10,i);
dx(5,i) = x(11,i);
dx(6,i) = x(12,i);
dx(7:9,i) = [0; 0; -g] + R*[0; 0; u(3,i)];
dx(10,i) = u(1,i);
dx(11,i) = u(2,i);
dx(12,i) = 0;

end


end