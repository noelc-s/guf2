function u = controller(t,x)
i=1;
g = 9.81;

phi = x(4,i);
theta = x(5,i);
psi = x(6,i);

Rz = @(t) [cos(t) sin(t) 0; -sin(t) cos(t) 0; 0 0 1];
Ry = @(t) [cos(t) 0 -sin(t); 0 1 0;sin(t) 0 cos(t)];
Rx = @(t) [1 0 0; 0 cos(t) sin(t); 0 -sin(t) cos(t)];

R = Rz(psi)*Ry(phi)*Rx(theta);

R_eul = [1 sin(phi)*tan(theta) cos(phi)*tan(theta); 0 cos(phi) -sin(phi); 0 sin(phi)*sec(theta) cos(phi)*sec(theta)];

wingbeat = 2;

% g_t = abs(sin(2*pi*wingbeat*t(i)))+eps;
g_t = 1;

f = [x(7:9,i); R_eul*x(10:12,i);[0; 0; -g];- cross(x(4:6,i),x(4:6,i))];
% g = [zeros(6,6); R*[g_t 0 0;0 g_t g_t/2; 0 0 g_t] zeros(3); [0 0 0 g_t 0 0;0 0 0 0 g_t 0;0 0 0 0 0 g_t]];
g = [zeros(6,6); R*[g_t 0 0;0 g_t 0; 0 0 g_t] zeros(3); [0 0 0 g_t 0 0;0 0 0 0 g_t 0;0 0 0 0 0 g_t]];

% dy_dx = [1 zeros(1,11); 0 1 zeros(1,10); 0 0 1 zeros(1,9); zeros(1,6) 1 zeros(1,5); zeros(1,7) 1 zeros(1,4); zeros(1,8) 1 zeros(1,3)];
dy_dx = eye(12);

Lfh = dy_dx*f;
Lgh = dy_dx*g;

KP = 100;
KD = 100;
A = [zeros(6) eye(6); -KP*eye(6) -KD*eye(6)];

u = (Lgh'*Lgh)\Lgh'*(-Lfh+A*dy_dx*x);
% u = [-KP*eye(6) -KD*eye(6)+[zeros(3,6);zeros(3,3) R_eul]]*x;

if any(isnan(u))
   u = zeros(6,1); 
end

end