% 2D Fluid Simulation in MATLAB
% Cleaned and commented version
% To increase performance decrease the number of Jacobi iterations, add less inflow
% particles or add them less often (e.g. only every 2nd simulation iteration)

% Simulation parameters
s = 200;                            % Grid size
ar = 2;                             % Aspect ratio
J = [0 1 0; 1 0 1; 0 1 0]/4;        % Stencil for Jacobi method

% Create a grid
[X, Y] = meshgrid(1:s*ar, 1:s);

% Initialize pressure and velocity fields
[p, vx, vy] = deal(zeros(s, s*ar));

% Initial positions of particles (spread over a larger area)
[px, py] = meshgrid(1:0.5:s*ar, 1:0.5:s);
px = reshape(px, numel(px), 1);
py = reshape(py, numel(py), 1);

% Save these initial positions for the inflow
pxo = px;
pyo = py;

f = figure(786); % Create figure to check if it's closed

% Main simulation loop (stops when closing figure window)
while ishandle(f)
    % Set initial velocity in a larger specific region
    vx(80:120, 5:25) = 1; % Increased velocity magnitude
    
    % Compute right-hand side for pressure equation
    rhs = -computeDivergence(vx, vy);
    
    % Jacobi iteration to solve for pressure
    % Higher number of iterations yields better solution
    for i = 1:100
        p = conv2(p, J, 'same') + rhs/4;
    end
    
    % Compute velocity gradient and update velocities for non-boundary pixels
    [dx, dy] = gradient(p);
    vx(2:end-1, 2:end-1) = vx(2:end-1, 2:end-1) - dx(2:end-1, 2:end-1);
    vy(2:end-1, 2:end-1) = vy(2:end-1, 2:end-1) - dy(2:end-1, 2:end-1);   
    
    % Advect velocity field using Runge-Kutta 4th order method (-1 = backward)
    [pvx, pvy] = RK4(X, Y, vx, vy, X, Y, -1);
    vx = interp2(X, Y, vx, pvx, pvy, 'linear', 0);
    vy = interp2(X, Y, vy, pvx, pvy, 'linear', 0);  
    
    % Advect particles using Runge-Kutta 4th order method (1 = forward)
    [px, py] = RK4(X, Y, vx, vy, px, py, 1);
    
    % Add the inflow particles
    % px = [px; pxo];
    % py = [py; pyo];
    
    % Visualization of particle positions
    scatter(px, py, 1, 'filled');
    axis equal; 
    axis([0 s*ar 0 s]);
    xlim([0, s*ar]);
    ylim([0, s]);
    
    % Temporary hide the grid lines
    grid off;

    drawnow;
    
    % after drawing, show grid lines again if needed
    grid on;
end

% Function for computing divergence
function div = computeDivergence(vx, vy)
    [dx_vx, ~] = gradient(vx);
    [~, dy_vy] = gradient(vy);
    div = dx_vx + dy_vy;
end

% Function for Runge-Kutta 4th order method for advection
function [x_new, y_new] = RK4(X, Y, vx, vy, px, py, h)
   k1x = interp2(X, Y, vx, px, py, 'linear', 0);
   k1y = interp2(X, Y, vy, px, py, 'linear', 0);
   k2x = interp2(X, Y, vx, px + h/2 * k1x, py + h/2 * k1y, 'linear', 0);
   k2y = interp2(X, Y, vy, px + h/2 * k1x, py + h/2 * k1y, 'linear', 0);
   k3x = interp2(X, Y, vx, px + h/2 * k2x, py + h/2 * k2y, 'linear', 0);
   k3y = interp2(X, Y, vy, px + h/2 * k2x, py + h/2 * k2y, 'linear', 0);
   k4x = interp2(X, Y, vx, px + h * k3x, py + h * k3y, 'linear', 0);
   k4y = interp2(X, Y, vy, px + h * k3x, py + h * k3y, 'linear', 0);
   x_new = px + h/6 * (k1x + 2*k2x + 2*k3x + k4x);
   y_new = py + h/6 * (k1y + 2*k2y + 2*k3y + k4y);
end
