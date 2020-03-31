function parameters=solve_sdp(parameters,x)
%[parameters] = solve_sdp(n_x, n_h, n_y, lambd, mu, Lip, ind_Lip, parameters)

addpath(genpath('..\YALMIP-master'))
addpath(genpath('C:\Program Files\Mosek\9.0'))

n_x = double(x.n_x);
n_h = double(x.n_h);
n_y = double(x.n_y);
lambd = double(x.rho);
mu = double(x.mu);
ind_Lip = int64(x.ind_Lip); % 1 Lipschitz regularizaion, 2 Enforcing Lipschitz bounds
Lip = double(x.L_des);

W0 = parameters.W0;
W1 = parameters.W1;

T = squeeze(x.T);

Y0 = parameters.Y0;
Y1 = parameters.Y1;

W0_bar = sdpvar(n_h, n_x);
W1_bar = sdpvar(n_y, n_h);

eps=10^(-9);
if ind_Lip==1
    rho = sdpvar(1);
    F1 = [[-rho*eye(n_x), W0_bar'*T, zeros(n_x,n_y); T*W0_bar, -2*T, ...
        W1_bar'; zeros(n_y,n_x),W1_bar, -eye(n_y)]<=-eps*eye(n_x+n_h+n_y)];
    optimize(F1,mu*rho+ norm(W0-W0_bar,'fro')^2 * (lambd/2) ...
        + norm(W1-W1_bar,'fro')^2 * (lambd/2) ...
        + trace(Y0'*(W0-W0_bar)) ...
        + trace(Y1'*(W1-W1_bar)));
elseif ind_Lip==2
    rho = Lip^2;
    F1 = [[-rho*eye(n_x), W0_bar'*T, zeros(n_x,n_y);  T*W0_bar, -2*T, ...
        W1_bar'; zeros(n_y,n_x),W1_bar, -eye(n_y)]<=-eps*eye(n_x+n_h+n_y)];
    optimize(F1,mu*rho+ norm(W0-W0_bar,'fro')^2 * (lambd/2) ...
        + norm(W1-W1_bar,'fro')^2 * (lambd/2) ...
        + trace(Y0'*(W0-W0_bar)) ...
        + trace(Y1'*(W1-W1_bar)));
end

Lipschitz = sqrt(value(rho));
W0_bar = value(W0_bar);
W1_bar = value(W1_bar);

Y0 = Y0 + lambd * (W0-W0_bar); % dual update step
Y1 = Y1 + lambd * (W1-W1_bar);

parameters.W0_bar = W0_bar;
parameters.W1_bar = W1_bar;

parameters.Y0 = Y0;
parameters.Y1 = Y1;

parameters.Lipschitz=Lipschitz;

