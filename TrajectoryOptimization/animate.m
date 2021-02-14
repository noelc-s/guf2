plot3(x(:,1),x(:,2),x(:,3))

hold on

l = 0.005;

f = quiver3(0,0,0,0,0,l,'r');

% axis equal

Rz = @(t) [cos(t) sin(t) 0; -sin(t) cos(t) 0; 0 0 1];
Ry = @(t) [cos(t) 0 -sin(t); 0 1 0;sin(t) 0 cos(t)];
Rx = @(t) [1 0 0; 0 cos(t) sin(t); 0 -sin(t) cos(t)];

wingbeat = 180;

tic
for i = 1:length(t)
    time = toc/10;
    t_now=find(t>=time,1);
%     u = interp1(traj.t,traj.u',t(i))';
    
    u = controller(t(t_now),x(t_now,:)');
    t1 = x(t_now,4);
    t2 = x(t_now,5);
    t3 = x(t_now,6);
    R = Rz(t1)*Ry(t2)*Rx(t3);
    u = R*[0;sin(2*pi*wingbeat*t(t_now))*u(3)/2; u(3)];
    u = u/norm(u);
    set(f,'XData',x(t_now,1),'YData',x(t_now,2),'ZData',x(t_now,3),...
        'UData',u(1),'VData',u(2),'WData',u(3))
    drawnow;
%     pause(0.05)
end