bettercolors

% figure;
% hold on
% %
width=3.4;
height=5.2/2;
x0 = 5;
y0 = 5;
fontsize = 10;
figure('Units','inches','Position',[x0 y0 width height],'PaperPositionMode','auto');

Fig1a = subplot(2,2,1);
% xlabel(Fig1a,{'$t$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
%
set(Fig1a,'Units','normalized','FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
ylabel(Fig1a,'Length','FontUnits','points','FontWeight','normal','FontSize',12,'FontName','Helvetica')

hold on
grid
% Fig1b.XLim = [0 5.5];
% Fig1b.YLim = [0 1.6];
Fig1a.Box = 'on';
Fig1a.XColor = 'k';
Fig1a.YColor = 'k';

set(gca,'LineWidth',1.5)

bvals = linspace(0.5,5,10);

bI = bvals(1:2);
bII = bvals(3);
bIII = bvals(8);
bIV = bvals(end);

for j = 1:length(bIII)
  bvalue = bIII(j)
  filename = [num2str(bvalue,'%.1f'),'_data.mat'];
  load(filename)

  is_polar = zeros(size(u,1),1);
  differences = zeros(size(u,1),1);
  color = zeros(size(u,1),3);
  for i = 1:size(u,1)
    is_polar(i) = is_polarized(u(i,:));
    differences(i) = max(u(i,:))-min(u(i,:));
    color(i,:) = (is_polar(i)>0)*bright(6,:)+(is_polar(i)==0)*bright(2,:);
  end

  active_u = zeros(size(u,1),1);
  for i = 1:size(u,1)
    active_u(i) = mean(u(i,:));
  end

  % color = color.*(bvalue/3);
  % scatter(bvalue*ones(size(l)),l,16,color,'filled')
  % scatter(bvalue+bvalue*t/35000,l,16,color,'filled')
  % scatter(active_u,l,8,color,'filled')
  scatter(Fig1a,t,l,10,color,'filled')
  xlim([0 2000])


  Fig1b = subplot(2,2,3);
  Fig1b.Box = 'on';
  Fig1b.XColor = 'k';
  Fig1b.YColor = 'k';
  set(gca,'LineWidth',1.5)
  set(Fig1b,'Units','normalized','FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
  hold on
  ylabel(Fig1b,{'max - min Rac'},'FontUnits','points','FontWeight','normal','FontSize',12,'FontName','Helvetica')
  xlabel(Fig1b,{'Time'},'FontUnits','points','FontWeight','normal','FontSize',12,'FontName','Helvetica')
  grid

  scatter(Fig1b,t,differences,10,color,'filled')
  xlim([0 2000])

end

Fig1c = subplot(2,2,2);
% xlabel(Fig1a,{'$t$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
% ylabel(Fig1a,{'$L$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
%
set(Fig1c,'Units','normalized','FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
hold on
grid
% Fig1b.XLim = [0 5.5];
% Fig1b.YLim = [0 1.6];
Fig1c.Box = 'on';
Fig1c.XColor = 'k';
Fig1c.YColor = 'k';
set(gca,'LineWidth',1.5)

for j = 1:length(bII)
  bvalue = bII(j)
  filename = [num2str(bvalue,'%.1f'),'_data.mat'];
  load(filename)

  is_polar = zeros(size(u,1),1);
  differences = zeros(size(u,1),1);
  color = zeros(size(u,1),3);
  for i = 1:size(u,1)
    is_polar(i) = is_polarized(u(i,:));
    differences(i) = max(u(i,:))-min(u(i,:));
    color(i,:) = (is_polar(i)>0)*bright(6,:)+(is_polar(i)==0)*bright(2,:);
  end

  active_u = zeros(size(u,1),1);
  for i = 1:size(u,1)
    active_u(i) = mean(u(i,:));
  end

  % color = color.*(bvalue/3);
  % scatter(bvalue*ones(size(l)),l,16,color,'filled')
  % scatter(bvalue+bvalue*t/35000,l,16,color,'filled')
  % scatter(active_u,l,8,color,'filled')
  scatter(Fig1c,t,l,10,color,'filled')
  xlim([0 2000])



  Fig1d = subplot(2,2,4);
  Fig1d.Box = 'on';
  Fig1d.XColor = 'k';
  Fig1d.YColor = 'k';
  set(gca,'LineWidth',1.5)
  set(Fig1d,'Units','normalized','FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')

  hold on
  % xlabel(Fig1d,{'Time'},'FontUnits','points','FontWeight','normal','FontSize',10,'FontName','Helvetica')
  grid

  scatter(Fig1d,t,differences,10,color,'filled')
  xlim([0 2000])

end

print(1,'Fig5b','-depsc','-painters')
