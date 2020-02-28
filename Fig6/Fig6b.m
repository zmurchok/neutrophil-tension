bettercolors
load('dataFig6b.mat')

% figure;
% hold on
% %
width=3.4/2;
height=5.2/2;
x0 = 5;
y0 = 5;
fontsize = 10;
figure('Units','inches','Position',[x0 y0 width height],'PaperPositionMode','auto');

Fig1a = subplot(2,1,1);
% xlabel(Fig1a,{'$t$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
%
set(Fig1a,'Units','normalized','FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
ylabel(Fig1a,'Area','FontUnits','points','FontWeight','normal','FontSize',12,'FontName','Helvetica')

hold on
grid
% Fig1b.XLim = [0 5.5];
% Fig1b.YLim = [0 1.6];
Fig1a.Box = 'on';
Fig1a.XColor = 'k';
Fig1a.YColor = 'k';

set(gca,'LineWidth',1.5)
idx = 1:1600;

t = timestep(idx)*0.01;
is_polar = zeros(length(idx),1);
color = zeros(length(idx),3);
for i = 1:length(idx)
  color(i,:) = (Rac_Max_Min(i)>0.15)*bright(6,:)+(Rac_Max_Min(i)<=0.15)*bright(2,:);
end
%
scatter(Fig1a,t,Area(idx),10,color,'filled')
xlim([0 10])
xticks([0 5 10])
ylim([1 1.3])
yticks([1 1.1 1.2 1.3])
yticklabels({'1', '', '', '1.3'})
xticklabels({'0', '', '10'})

Fig1b = subplot(2,1,2);
Fig1b.Box = 'on';
Fig1b.XColor = 'k';
Fig1b.YColor = 'k';
set(gca,'LineWidth',1.5)
set(Fig1b,'Units','normalized','FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Helvetica')
hold on
ylabel(Fig1b,{'max - min Rac'},'FontUnits','points','FontWeight','normal','FontSize',12,'FontName','Helvetica')
xlabel(Fig1b,{'Time'},'FontUnits','points','FontWeight','normal','FontSize',12,'FontName','Helvetica')
grid

scatter(Fig1b,t,Rac_Max_Min(idx),10,color,'filled')
xlim([0 10])
ylim([0 0.8])
yticks([0 0.4 0.8])
xticks([0 5 10])
yticklabels({'0', '', '0.8'})
xticklabels({'0', '', '10'})


print(1,'Fig6b.eps','-depsc','-painters')
