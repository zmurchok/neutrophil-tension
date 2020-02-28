bettercolors;

global cds sys

sys.gui.pausespecial=0;  %Pause at special points
sys.gui.pausenever=1;    %Pause never
sys.gui.pauseeachpoint=0; %Pause at each point

syshandle=@funcs;  %Specify system file

SubFunHandles=feval(syshandle);  %Get function handles from system file
RHShandle=SubFunHandles{2};      %Get function handle for ODE

b = 0.1;
L= 0.9;
RT=2;
delta0=3;
delta1=1;
l0=1;
gma=5;
n=6;
V=0;

local = 1;
codim = 0;

xinit=[0;0]; %Set ODE initial condition

%Specify ODE function with ODE parameters set
RHS_no_param=@(t,x)RHShandle(t,x,b,L,gma,n,RT,delta0,delta1,l0,V);

%Set ODE integrator parameters.
options=odeset;
options=odeset(options,'RelTol',1e-12);
options=odeset(options,'AbsTol',1e-12);

%Integrate until a steady state is found.
[tout xout]=ode45(RHS_no_param,[0,200],xinit,options);

xinit=xout(size(xout,1),:);

pvec=[b,L,gma,n,RT,delta0,delta1,l0,V]';      % Initialize parameter vector

ap=1;

[x0,v0]=init_EP_EP(syshandle, xinit', pvec, ap); %Initialize equilibrium


opt=contset;
opt=contset(opt,'MaxNumPoints',500); %Set numeber of continuation steps
opt=contset(opt,'MaxStepsize',0.1);  %Set max step size
opt=contset(opt,'Singularities',1);  %Monitor singularities
opt=contset(opt,'Eigenvalues',1);    %Output eigenvalues
opt=contset(opt,'InitStepsize',0.01); %Set Initial stepsize

[x1,v1,s1,h1,f1]=cont(@equilibrium,x0,v0,opt);

opt=contset(opt,'backward',1);
[x1b,v1b,s1b,h1b,f1b]=cont(@equilibrium,x0,v0,opt);


if local == 1
%%%% LOCAL BRANCHES

ind = 2;
xBP=x1(1:2,s1(ind).index);       %Extract branch point
pvec(ap)=x1(3,s1(ind).index);  %Extract branch point
[x0,v0]=init_BP_EP(syshandle, xBP, pvec, s1(ind), 0.05);
% opt=contset(opt,'backward',1);
% [x3b,v3b,s3b,h3b,f3b]=cont(@equilibrium,x0,v0,opt); %Switch branches and continue.
opt = contset(opt,'backward',0);
[x3,v3,s3,h3,f3]=cont(@equilibrium,x0,v0,opt); %Switch branches and continue.

% ind = 3;
% xBP=x1(1:2,s1(ind).index);       %Extract branch point
% pvec(ap)=x1(3,s1(ind).index);  %Extract branch point
% [x0,v0]=init_BP_EP(syshandle, xBP, pvec, s1(ind), 0.001);
% % opt = contset(opt,'backward',1);
% % [x4b,v4b,s4b,h4b,f4b]=cont(@equilibrium,x0,v0,opt); %Switch branches and continue.
% opt = contset(opt,'backward',0);
% [x4,v4,s4,h4,f4]=cont(@equilibrium,x0,v0,opt); %Switch branches and continue.

end
% CODIM 1 plots


width=5.2;
height=5.2/2;
x0 = 5;
y0 = 5;
fontsize = 10;
f = figure('Units','inches','Position',[x0 y0 width height],'PaperPositionMode','auto');
Fig2a = subplot(1,2,1);
xlabel(Fig2a,{'$b$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
ylabel(Fig2a,{'$R^\ell$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
Fig2aTitle = title(Fig2a,{'(a)'},'FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica');
set(Fig2a,'Units','normalized','FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
grid
Fig2a.XLim = [0 6];
Fig2a.YLim = [0 2];
Fig2a.Box = 'on';
set(gca,'LineWidth',1.5)
hold on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GLOBAL curves
curves = {'1','1b'};

for i = 1:length(curves)
  xeqcurve=eval(['x' curves{i}]);
  minevaleq=eval(['f' curves{i} '(2,:)']); %This is the last eigenvalue.  That is the one that determines stability

  L=length(xeqcurve(1,:));

  curveind=1;
  lengthind=0;
  maxlengthind=0;
  evalstart=floor(heaviside(minevaleq(1)));
  datamateq=zeros(4,L);

  for i=1:L
    evalind=floor(heaviside(minevaleq(i)));
    if evalstart~=evalind
        curveind=curveind+1;
        i;
        evalstart=evalind;
        maxlengthind=max(lengthind,maxlengthind);
        lengthind=0;
    end
    datamateq(1,i)=xeqcurve(3,i); % This is the parameter that is varied.
    datamateq(2,i)=xeqcurve(1,i); % This is the dependent axis of the bifurcation plot.  The one you wish to plot
    datamateq(3,i)=evalind;
    datamateq(4,i)=curveind;

    lengthind=lengthind+1;
  end

  maxlengthind=max(maxlengthind,lengthind);

  curveindeq=curveind;

  for i=1:curveindeq
    index=find(datamateq(4,:)==i);
    eval(['curve' num2str(i) 'eq' '=datamateq(1:3,index);']);
  end

  for i=1:curveindeq
    stability=eval(['curve' num2str(i) 'eq(3,1)']);
    if stability==0
        plotsty='-';
    else
        plotsty=':';
    end

    plotcolor='k';

    plotstr=strcat(plotcolor,plotsty);

    plot(eval(['curve' num2str(i) 'eq(1,:)']),eval(['curve' num2str(i) 'eq(2,:)']),plotstr,'Linewidth',3)
    hold on
  end
end

if local == 1
%%%LOCAL
curves = {'3'};

for i = 1:length(curves)
  xeqcurve=eval(['x' curves{i}]);
  minevaleq=eval(['f' curves{i} '(2,:)']); %This is the last eigenvalue.  That is the one that determines stability

  L=length(xeqcurve(1,:));

  curveind=1;
  lengthind=0;
  maxlengthind=0;
  evalstart=floor(heaviside(minevaleq(1)));
  datamateq=zeros(4,L);

  for i=1:L
    evalind=floor(heaviside(minevaleq(i)));
    if evalstart~=evalind
        curveind=curveind+1;
        i;
        evalstart=evalind;
        maxlengthind=max(lengthind,maxlengthind);
        lengthind=0;
    end
    datamateq(1,i)=xeqcurve(3,i); % This is the parameter that is varied.
    datamateq(2,i)=xeqcurve(2,i); % This is the dependent axis of the bifurcation plot.  The one you wish to plot
    datamateq(3,i)=evalind;
    datamateq(4,i)=curveind;

    lengthind=lengthind+1;
  end

  maxlengthind=max(maxlengthind,lengthind);

  curveindeq=curveind;

  for i=1:curveindeq
    index=find(datamateq(4,:)==i);
    eval(['curve' num2str(i) 'eq' '=datamateq(1:3,index);']);
  end

  for i=1:curveindeq
    stability=eval(['curve' num2str(i) 'eq(3,1)']);
    if stability==0
        plotsty='-';
    else
        plotsty=':';
    end

    mycolor = highcontrast(2,:);
    plotstr=plotsty;

    plot(eval(['curve' num2str(i) 'eq(1,:)']),eval(['curve' num2str(i) 'eq(2,:)']),plotstr,'Color',mycolor,'Linewidth',2)
    hold on
  end
end

end

opt=contset;
opt=contset(opt,'MaxNumPoints',5500); %Set numeber of continuation steps
opt=contset(opt,'MaxStepsize',1);  %Set max step size
opt=contset(opt,'Singularities',1);  %Monitor singularities
opt=contset(opt,'Eigenvalues',1);    %Output eigenvalues
opt=contset(opt,'InitStepsize',0.5); %Set Initial stepsize
aps = [1,2];

%
% ind = 2
% xtmp = x3(1:end-1,s3(ind).index);
% pvec(ap) = x3(end,s3(ind).index);
% [x0,v0] = init_LP_LP(syshandle,xtmp,pvec,aps);
% opt = contset(opt,'backward',0);
% [x5,v5,s5,h5,f5] = cont(@limitpoint,x0,v0,opt);
% opt = contset(opt,'backward',1);
% [x5b,v5b,s5b,h5b,f5b] = cont(@limitpoint,x0,v0,opt);

% ind = 3
% xtmp = x3(1:end-1,s3(ind).index);
% pvec(ap) = x3(end,s3(ind).index);
% [x0,v0] = init_LP_LP(syshandle,xtmp,pvec,aps);
% opt = contset(opt,'backward',0);
% [x6,v6,s6,h6,f6] = cont(@limitpoint,x0,v0,opt);
%   opt = contset(opt,'backward',1);
% [x6b,v6b,s6b,h6b,f6b] = cont(@limitpoint,x0,v0,opt);

ind = 4
xtmp = x3(1:end-1,s3(ind).index);
pvec(ap) = x3(end,s3(ind).index);
[x0,v0] = init_LP_LP(syshandle,xtmp,pvec,aps);
opt = contset(opt,'backward',0);
[x7,v7,s7,h7,f7] = cont(@limitpoint,x0,v0,opt);
  opt = contset(opt,'backward',1);
[x7b,v7b,s7b,h7b,f7b] = cont(@limitpoint,x0,v0,opt);

% ind = 5;
% xtmp = x3(1:end-1,s3(ind).index);
% pvec(ap) = x3(end,s3(ind).index);
% [x0,v0] = init_LP_LP(syshandle,xtmp,pvec,aps);
% opt = contset(opt,'backward',0);
% [x8,v8,s8,h8,f8] = cont(@limitpoint,x0,v0,opt);
%   opt = contset(opt,'backward',1);
% [x8b,v8b,s8b,h8b,f8b] = cont(@limitpoint,x0,v0,opt);
%
% ind = 6;
% xtmp = x3(1:end-1,s3(ind).index);
% pvec(ap) = x3(end,s3(ind).index);
% [x0,v0] = init_LP_LP(syshandle,xtmp,pvec,aps);
% opt = contset(opt,'backward',0);
% [x9,v9,s9,h9,f9] = cont(@limitpoint,x0,v0,opt);
%   opt = contset(opt,'backward',1);
% [x9b,v9b,s9b,h9b,f9b] = cont(@limitpoint,x0,v0,opt);
%
% ind = 7;
% xtmp = x3(1:end-1,s3(ind).index);
% pvec(ap) = x3(end,s3(ind).index);
% [x0,v0] = init_LP_LP(syshandle,xtmp,pvec,aps);
% opt = contset(opt,'backward',0);
% [x10,v10,s10,h10,f10] = cont(@limitpoint,x0,v0,opt);
%   opt = contset(opt,'backward',1);
% [x10b,v10b,s10b,h10b,f10b] = cont(@limitpoint,x0,v0,opt);


%BPs
ind = 2
xtmp = x1(1:end-1,s1(ind).index);
pvec(ap) = x1(end,s1(ind).index);
[x0,v0] = init_BP_LP(syshandle,xtmp,pvec,aps);
opt = contset(opt,'backward',0);
[x11,v11,s11,h11,f11] = cont(@limitpoint,x0,v0,opt);
opt = contset(opt,'backward',1);
[x11b,v11b,s11b,h11b,f11b] = cont(@limitpoint,x0,v0,opt);

ind = 7
xtmp = x1(1:end-1,s1(ind).index);
pvec(ap) = x1(end,s1(ind).index);
[x0,v0] = init_BP_LP(syshandle,xtmp,pvec,aps);
opt = contset(opt,'backward',0);
[x12,v12,s12,h12,f12] = cont(@limitpoint,x0,v0,opt);
opt = contset(opt,'backward',1);
[x12b,v12b,s12b,h12b,f12b] = cont(@limitpoint,x0,v0,opt);

% ind = 5;
% xtmp = x3(1:end-1,s3(ind).index);
% pvec(ap) = x3(end,s3(ind).index);
% [x0,v0] = init_BP_LP(syshandle,xtmp,pvec,aps);
% opt = contset(opt,'backward',0);
% [x13,v13,s13,h13,f13] = cont(@limitpoint,x0,v0,opt);
% opt = contset(opt,'backward',1);
% [x13b,v13b,s13b,h13b,f13b] = cont(@limitpoint,x0,v0,opt);

%%%
close all

width=5.2;
height=5.2/2;
x0 = 5;
y0 = 5;
fontsize = 10;
figure('Units','inches','Position',[x0 y0 width/2 height],'PaperPositionMode','auto');
Fig5a = subplot(1,1,1);
xlabel(Fig5a,{'$b$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',12,'FontName','Helvetica')

set(Fig5a,'Units','normalized','FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
hold on
grid
Fig5a.XLim = [0 5.5];
Fig5a.YLim = [0 2.2];
Fig5a.Box = 'on';
Fig5a.XColor = 'k';
Fig5a.YColor = 'k';
ylabel({'$L$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',12,'FontName','Helvetica')
set(gca,'LineWidth',1.5)
curves = {'5','6','7','8'};

linecolor(5) = {highcontrast(2,:)};
linecolor(6) = {highcontrast(2,:)};
linecolor(7) = {highcontrast(2,:)};
linecolor(8) = {highcontrast(2,:)};
linecolor(9) = {highcontrast(2,:)};
linecolor(10) = {highcontrast(2,:)};
linecolor(11) = {highcontrast(3,:)};
linecolor(12) = {highcontrast(3,:)};
linecolor(13) = {highcontrast(3,:)};

linestyle(5) = {'-'};
linestyle(6) = {'-'};
linestyle(7) = {'-'};
linestyle(8) = {'-'};
linestyle(9) = {'-'};
linestyle(10) = {'-'};
linestyle(11) = {'--'};
linestyle(12) = {'--'};
linestyle(13) = {'--'};

for i = [7,11,12]
  i
  xeqcurve=eval(['x' num2str(i)]);
  plot(xeqcurve(end-1,:),xeqcurve(end,:),'LineStyle',linestyle{i},'Color',linecolor{i},'LineWidth',4)
  xeqcurve = eval(['x' num2str(i) 'b']);
  plot(xeqcurve(end-1,:),xeqcurve(end,:),'LineStyle',linestyle{i},'Color',linecolor{i},'LineWidth',4)
end

% scatter(0.1,7,'ko','filled')
% text(0.1,7,' (b)')
%
% scatter(0.1,3,'ko','filled')
% text(0.1,3,' (c)')
%
% scatter(4.5,6,'ko','filled')
% text(4.5,6,' (d)')
%
% scatter(4.5,3,'ko','filled')
% text(4.5,3,' (e)')



% print(1,'1a','-depsc')
% print(1,'Fig5a','-depsc','-painters')
savefig(1,'RT_2.fig')
