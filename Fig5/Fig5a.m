%FIRST DO THIS
% load('/Users/zmurchc/Dropbox/Apps/Overleaf/neutrophil-tension-paper/code/revisions/Fig5/xppdat.mat')
% plotxppaut4p4
%PLOT Fig5LPA.dat
%SAVE THIS FIGURE


close all
clear all


bettercolors
uiopen('/Users/zmurchc/Dropbox/Apps/Overleaf/neutrophil-tension-paper/code/revisions/Fig5/Fig5LPA.fig',1)



% pause
%
width=5.2;
height=5.2/2;
x0 = 5;
y0 = 5;
fontsize = 12;
set(gcf,'Units','inches','Position',[x0 y0 width/2 height],'PaperPositionMode','auto')
% figure('Units','inches','Position',[x0 y0 width/2 height],'PaperPositionMode','auto');
% Fig1b = subplot(1,1,1);
xlabel({'$b$'},'Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Times')
ylabel({'$L$'},'Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Times')

% set(gca,'Units','normalized','FontWeight','normal','FontSize',fontsize,'FontName','Times')
hold on
grid
xlim([0 5.5])
ylim([0.96 1.07])
% Fig1b.Box = 'on';
set(gca,'LineWidth',1.5)
set(gca,'Box','on')
legend('off')

bvals = linspace(0.5,5,10);
for j = 1:length(bvals)
  bvalue = bvals(j)
  filename = [num2str(bvalue,'%.1f'),'_data.mat'];
  load(filename)

  is_polar = zeros(size(u,1),1);
  color = zeros(size(u,1),3);
  for i = 1:size(u,1)
    is_polar(i) = is_polarized(u(i,:));
    color(i,:) = (is_polar(i)>0)*bright(6,:)+(is_polar(i)==0)*bright(2,:);
  end

  active_u = zeros(size(u,1),1);
  for i = 1:size(u,1)
    active_u(i) = mean(u(i,:));
  end

  % color = color.*(bvalue/3);
  scatter(bvalue*ones(size(l)),l,16,color,'filled')
  % scatter(bvalue+bvalue*t/35000,l,16,color,'filled')
  % scatter(active_u,l,8,color,'filled')
  % scatter(t,l,16,color,'filled')
  % ylim([0.98 1.37])
  % xlim([0 1.4])
end
% ylim([0 1.85])

% print(1,'Fig5a','-depsc','-painters')
