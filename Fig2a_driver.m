bettercolors;

global cds sys

sys.gui.pausespecial=0;  %Pause at special points
sys.gui.pausenever=1;    %Pause never
sys.gui.pauseeachpoint=0; %Pause at each point

syshandle=@Fig2_Functions;  %Specify system file

SubFunHandles=feval(syshandle);  %Get function handles from system file
RHShandle=SubFunHandles{2};      %Get function handle for ODE

b = 0.01;
gma = 5;
n = 6;
RT = 2;
delta = 2;

local = 1;
codim = 1;

xinit=[0;0]; %Set ODE initial condition

%Specify ODE function with ODE parameters set
RHS_no_param=@(t,x)RHShandle(t,x,b,gma,n,RT,delta);

%Set ODE integrator parameters.
options=odeset;
options=odeset(options,'RelTol',1e-8);
options=odeset(options,'maxstep',1e-2);

%Integrate until a steady state is found.
[tout xout]=ode45(RHS_no_param,[0,200],xinit,options);

xinit=xout(size(xout,1),:);

pvec=[b,gma,n,RT,delta]';      % Initialize parameter vector

ap=1;

[x0,v0]=init_EP_EP(syshandle, xinit', pvec, ap); %Initialize equilibrium


opt=contset;
opt=contset(opt,'MaxNumPoints',1500); %Set numeber of continuation steps
opt=contset(opt,'MaxStepsize',.01);  %Set max step size
opt=contset(opt,'Singularities',1);  %Monitor singularities
opt=contset(opt,'Eigenvalues',1);    %Output eigenvalues
opt=contset(opt,'InitStepsize',0.01); %Set Initial stepsize

[x1,v1,s1,h1,f1]=cont(@equilibrium,x0,v0,opt);

opt=contset(opt,'backward',1);
[x1b,v1b,s1b,h1b,f1b]=cont(@equilibrium,x0,v0,opt);


if local == 1
%% LOCAL BRANCHES

ind = 2;
xBP=x1(1:2,s1(ind).index);       %Extract branch point
pvec(ap)=x1(3,s1(ind).index);  %Extract branch point
[x0,v0]=init_BP_EP(syshandle, xBP, pvec, s1(ind), 0.05);
opt=contset(opt,'backward',1);
[x3b,v3b,s3b,h3b,f3b]=cont(@equilibrium,x0,v0,opt); %Switch branches and continue.
opt = contset(opt,'backward',0);
[x3,v3,s3,h3,f3]=cont(@equilibrium,x0,v0,opt); %Switch branches and continue.

ind = 7;
xBP=x1(1:2,s1(ind).index);       %Extract branch point
pvec(ap)=x1(3,s1(ind).index);  %Extract branch point
[x0,v0]=init_BP_EP(syshandle, xBP, pvec, s1(ind), 0.05);
opt = contset(opt,'backward',1);
[x4b,v4b,s4b,h4b,f4b]=cont(@equilibrium,x0,v0,opt); %Switch branches and continue.
opt = contset(opt,'backward',0);
[x4,v4,s4,h4,f4]=cont(@equilibrium,x0,v0,opt); %Switch branches and continue.

end
% CODIM 1 plots


width=5.2/2;
height=5.2/2;
x0 = 5;
y0 = 5;
fontsize = 10;
f = figure('Units','inches','Position',[x0 y0 width height],'PaperPositionMode','auto');
Fig2a = subplot(1,1,1);
xlabel(Fig2a,{'$b$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
ylabel(Fig2a,{'$R^\ell$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
Fig2aTitle = title(Fig2a,{'(a)'},'FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica');
set(Fig2a,'Units','normalized','FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
grid
Fig2a.XLim = [0 4];
Fig2a.YLim = [0 5];
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
curves = {'3','3b'};

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
aps = [1,5];
%%%WP
ind = 4;
xtmp = x3(1:end-1,s3(ind).index);
pvec(ap) = x3(end,s3(ind).index);
[x0,v0] = init_LP_LP(syshandle,xtmp,pvec,aps);
opt = contset(opt,'backward',0);
[x5,v5,s5,h5,f5] = cont(@limitpoint,x0,v0,opt);
opt = contset(opt,'backward',1);
[x5b,v5b,s5b,h5b,f5b] = cont(@limitpoint,x0,v0,opt);

ind = 5
xtmp = x3b(1:end-1,s3b(ind).index);
pvec(ap) = x3b(end,s3b(ind).index);
[x0,v0] = init_LP_LP(syshandle,xtmp,pvec,aps);
opt = contset(opt,'backward',0);
[x6,v6,s6,h6,f6] = cont(@limitpoint,x0,v0,opt);
  opt = contset(opt,'backward',1);
[x6b,v6b,s6b,h6b,f6b] = cont(@limitpoint,x0,v0,opt);

%BPs
ind = 2;
xtmp = x3b(1:end-1,s3b(ind).index);
pvec(ap) = x3b(end,s3b(ind).index);
[x0,v0] = init_BP_LP(syshandle,xtmp,pvec,aps);
opt = contset(opt,'backward',0);
[x7,v7,s7,h7,f7] = cont(@limitpoint,x0,v0,opt);
opt = contset(opt,'backward',1);
[x7b,v7b,s7b,h7b,f7b] = cont(@limitpoint,x0,v0,opt);

ind = 8;
xtmp = x3b(1:end-1,s3b(ind).index);
pvec(ap) = x3b(end,s3b(ind).index);
[x0,v0] = init_BP_LP(syshandle,xtmp,pvec,aps);
opt = contset(opt,'backward',0);
[x8,v8,s8,h8,f8] = cont(@limitpoint,x0,v0,opt);
opt = contset(opt,'backward',1);
[x8b,v8b,s8b,h8b,f8b] = cont(@limitpoint,x0,v0,opt);

%%%
close 1

width=5.2/2;
height=5.2/2;
x0 = 5;
y0 = 5;
fontsize = 10;

figure('Units','inches','Position',[x0 y0 width height],'PaperPositionMode','auto');

Fig1b = subplot(1,1,1);
xlabel(Fig1b,{'$b$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',12,'FontName','Helvetica','color','k')

% Fig1bTitle = title(Fig1b,{'(b)'},'FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica');
set(Fig1b,'Units','normalized','FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
hold on
grid
Fig1b.XLim = [0 6];
Fig1b.YLim = [0 10];
Fig1b.Box = 'on';
Fig1b.XColor = 'k';
Fig1b.YColor =  'k';
ylabel({'$\delta$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',12,'FontName','Helvetica','color','k')
set(gca,'LineWidth',1.5)
curves = {'5','6','7','8'};

linecolor(5) = {highcontrast(2,:)};
linecolor(6) = {highcontrast(2,:)};
linecolor(7) = {highcontrast(3,:)};
linecolor(8) = {highcontrast(3,:)};


linestyle(5) = {'-'};
linestyle(6) = {'-'};
linestyle(7) = {'--'};
linestyle(8) = {'--'};

for i = 5:8
  xeqcurve=eval(['x' num2str(i)]);
  plot(xeqcurve(end-1,:),xeqcurve(end,:),'LineStyle',linestyle{i},'Color',linecolor{i},'LineWidth',4)
  xeqcurve = eval(['x' num2str(i) 'b']);
  plot(xeqcurve(end-1,:),xeqcurve(end,:),'LineStyle',linestyle{i},'Color',linecolor{i},'LineWidth',4)
end

scatter(0.1,7.5,'ko','filled')
text(0.1,7.5,' (b)','FontSize',fontsize,'FontName','Helvetica')

scatter(0.1,3,'ko','filled')
text(0.1,3,' (d)','FontSize',fontsize,'FontName','Helvetica')

scatter(4.5,3,'ko','filled')
text(4.5,3,' (e)','FontSize',fontsize,'FontName','Helvetica')

scatter(4.5,7.5,'ko','filled')
text(4.5,7.5,' (c)','FontSize',fontsize,'FontName','Helvetica')


print(1,'2a','-depsc')
