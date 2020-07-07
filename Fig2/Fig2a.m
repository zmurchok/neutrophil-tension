close all
clear all


bettercolors
uiopen('/Users/zmurchc/Google Drive/Research/2020/neutrophil-tension/revisions/Fig2/Fig2a.fig',1)

width=5.2;
height=5.2/2;
x0 = 5;
y0 = 5;
fontsize = 12;
set(gcf,'Units','inches','Position',[x0 y0 width/2 height],'PaperPositionMode','auto')
% figure('Units','inches','Position',[x0 y0 width/2 height],'PaperPositionMode','auto');
% Fig1b = subplot(1,1,1);
xlabel({'$b$'},'Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Times')
ylabel({'$\delta$'},'Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Times')

% set(gca,'Units','normalized','FontWeight','normal','FontSize',fontsize,'FontName','Times')
hold on
grid
xlim([0 6])
ylim([0 10])
% Fig1b.Box = 'on';
set(gca,'LineWidth',1.5)
set(gca,'Box','on')
set(gca,'XColor',[0 0 0])
set(gca,'YColor',[0 0 0])
legend('off')
scatter(0.1,7.5,'ko','filled')
text(0.1,7.5,' B','FontSize',fontsize,'FontName','Helvetica')

scatter(0.1,3,'ko','filled')
text(0.1,3,' D','FontSize',fontsize,'FontName','Helvetica')

scatter(4.5,3,'ko','filled')
text(4.5,3,' E','FontSize',fontsize,'FontName','Helvetica')

scatter(4.5,7.5,'ko','filled')
text(4.5,7.5,' C','FontSize',fontsize,'FontName','Helvetica')


print(1,'Fig2a','-depsc')
