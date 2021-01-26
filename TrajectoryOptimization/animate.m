plot3(x(:,1),x(:,2),x(:,3))

hold on

l = 0.005;

f = quiver3(0,0,0,0,0,l,'r');

axis equal

Rz = @(t) [cos(t) sin(t) 0; -sin(t) cos(t) 0; 0 0 1];
Ry = @(t) [cos(t) 0 -sin(t); 0 1 0;sin(t) 0 cos(t)];
Rx = @(t) [1 0 0; 0 cos(t) sin(t); 0 -sin(t) cos(t)];

for i = 1:length(t)
    u = interp1(traj.t,traj.u',t(i))';
    t1 = x(i,4);
    t2 = x(i,5);
    t3 = x(i,6);
    R = Rz(t1)*Ry(t2)*Rx(t3);
    u = R*[0; 0; u(3)*l];
    set(f,'XData',x(i,1),'YData',x(i,2),'ZData',x(i,3),...
        'UData',u(1),'VData',u(2),'WData',u(3))
    drawnow;
    pause(0.05)
end