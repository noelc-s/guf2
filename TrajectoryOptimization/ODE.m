function dx = ODE(t,x,u)

% drone EOM: https://towardsdatascience.com/demystifying-drone-dynamics-ee98b1ba882f

g = 9.81;
dx = x;

for i = 1:size(x,2)

phi = x(4,i);
theta = x(5,i);
psi = x(6,i);

Rz = @(t) [cos(t) sin(t) 0; -sin(t) cos(t) 0; 0 0 1];
Ry = @(t) [cos(t) 0 -sin(t); 0 1 0;sin(t) 0 cos(t)];
Rx = @(t) [1 0 0; 0 cos(t) sin(t); 0 -sin(t) cos(t)];

R = Rz(psi)*Ry(phi)*Rx(theta);

R_eul = [1 sin(phi)*tan(theta) cos(phi)*tan(theta); 0 cos(phi) -sin(phi); 0 sin(phi)*sec(theta) cos(phi)*sec(theta)];

dx(1,i) = x(7,i);
dx(2,i) = x(8,i);
dx(3,i) = x(9,i);
dx(4:6,i) = R_eul*x(10:12,i);
dx(7:9,i) = [0; 0; -g] + R*[0; 0; u(3,i)];
dx(10:12,i) = [u(1,i); u(2,i); 0] - cross(x(4:6,i),x(4:6,i));

end


end