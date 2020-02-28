function out = Fig1_Functions
out{1} = @init;
out{2} = @fun_eval;
out{3} = [];
out{4} = [];
out{5} = [];
out{6} = [];
out{7} = [];
out{8} = [];
out{9} = [];

% --------------------------------------------------------------------------
function dydt = fun_eval(t,y,b,gma,n,RT,delta)
dydt=[(b+gma*y(1)^n/(1+y(1)^n))*(RT-y(1)) - delta*y(1);(b+gma*y(2)^n/(1+y(2)^n))*(RT-y(1)) - delta*y(2);];

% --------------------------------------------------------------------------
function [tspan,y0,options] = init
handles = feval(PosFeedMutInhib);
y0=[0];
options = odeset('Jacobian',[],'JacobianP',[],'Hessians',[],'HessiansP',[]);
tspan = [0 10];

% --------------------------------------------------------------------------
function jac = jacobian(t,y,b,gma,n,RT,delta)
% --------------------------------------------------------------------------
function jacp = jacobianp(t,y,b,gma,n,RT,delta)
% --------------------------------------------------------------------------
function hess = hessians(t,y,b,gma,n,RT,delta)
% --------------------------------------------------------------------------
function hessp = hessiansp(t,y,b,gma,n,RT,delta)
%---------------------------------------------------------------------------
function Tension3  = der3(t,y,b,gma,n,RT,delta)
%---------------------------------------------------------------------------
function Tension4  = der4(t,y,b,gma,n,RT,delta)
%---------------------------------------------------------------------------
function Tension5  = der5(t,y,b,gma,n,RT,delta)
