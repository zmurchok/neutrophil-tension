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

function y = A(R,b,gma,n)
  y = b+gma*R^n/(1+R^n);

function y = delta(L,delta0,delta1,l0)
  y = delta0 + delta1 * (L-l0);

function y = f(L,V)
  y = 1; %L/(L+V);


% --------------------------------------------------------------------------
function dydt = fun_eval(t,y,b,L,gma,n,RT,delta0,delta1,l0,V)
dydt=[A(y(1),b,gma,n)*f(L,V)*(RT/L-y(1)) - delta(L,delta0,delta1,l0)*y(1);A(y(2),b,gma,n)*f(L,V)*(RT/L-y(1)) - delta(L,delta0,delta1,l0)*y(2);];

% --------------------------------------------------------------------------
function [tspan,y0,options] = init
handles = feval(PosFeedMutInhib);
y0=[0];
options = odeset('Jacobian',[],'JacobianP',[],'Hessians',[],'HessiansP',[]);
tspan = [0 10];

% --------------------------------------------------------------------------
function jac = jacobian(t,y,b,L,gma,n,RT,delta0,delta1,l0,V)
% --------------------------------------------------------------------------
function jacp = jacobianp(t,y,b,L,gma,n,RT,delta0,delta1,l0,V)
% --------------------------------------------------------------------------
function hess = hessians(t,y,b,L,gma,n,RT,delta0,delta1,l0,V)
% --------------------------------------------------------------------------
function hessp = hessiansp(t,y,b,L,gma,n,RT,delta0,delta1,l0,V)
%---------------------------------------------------------------------------
function Tension3  = der3(t,y,b,L,gma,n,RT,delta0,delta1,l0,V)
%---------------------------------------------------------------------------
function Tension4  = der4(t,y,b,L,gma,n,RT,delta0,delta1,l0,V)
%---------------------------------------------------------------------------
function Tension5  = der5(t,y,b,L,gma,n,RT,delta0,delta1,l0,V)
