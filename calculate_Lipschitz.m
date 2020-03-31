function Lip=calculate_Lipschitz(parameters,x)

addpath(genpath('..\YALMIP-master'))
addpath(genpath('C:\Program Files\Mosek\9.0'))
 
W0 = parameters.W0;
W1 = parameters.W1;
b0 = parameters.b0';
b1 = parameters.b1';

n_x = double(x.n_x);
n_h = double(x.n_h);
n_y = double(x.n_y);

T = diag(sdpvar(n_h,1));
rho = sdpvar(1); % rho=L^2

eps=10^(-9);
F1 = ([-rho*eye(n_x), W0'*T; T*W0, -2*T+W1'*W1]<=-eps*eye(n_x+n_h));

optimize(F1,rho);
Lip.Lipschitz=sqrt(double(rho));
Lip.T = double(T);

L=Lip.Lipschitz;
T=Lip.T;

% Check if incremental quadratic inequalities and LMIs are fulfilled
res_LMI=eig([-L^2*eye(n_x), W0'*T; T*W0, -2*T+W1'*W1]);
for j=1:10000
        x1=randn;
        x2=randn;
        vec1=x1-x2;
        vec2=tanh(W0*x1+b0)-tanh(W0*x2+b0);
    res_Lip(j)=vec2'*W1'*W1*vec2-L^2*vec1^2;
    res_sr(j)=[vec1;vec2]'*[0, W0'*T;T*W0, -2*T]*[vec1;vec2];
end

if min(res_sr)<0 % Slope restriction incremental quadratic inequality ok? 
    ok(1)=0;
else
    ok(1)=1;
end
if max(res_LMI)>0 % enforced LMI ok? 
    ok(2)=max(res_LMI);
else
    ok(2)=1;
end
if max(res_Lip)>0 % Lipschitz inequality ok? 
    ok(3)=0;
else
    ok(3)=1;
end

Lip.ok=ok;

