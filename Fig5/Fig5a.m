bettercolors
uiopen('/Users/zmurchc/Dropbox/Apps/Overleaf/neutrophil-tension-paper/code/Fig5/RT_2.fig',1)
% figure;
% hold on
% %
% width=5.2;
% height=5.2/2;
% x0 = 5;
% y0 = 5;
% fontsize = 12;
% figure('Units','inches','Position',[x0 y0 width/2 height],'PaperPositionMode','auto');
% Fig1b = subplot(1,1,1);
% % xlabel(Fig1b,{'$b$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Times')
% % ylabel(Fig1b,{'$b$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Times')
%
% set(Fig1b,'Units','normalized','FontUnits','points','FontWeight','normal','FontSize',fontsize,'FontName','Times')
% hold on
% grid
% Fig1b.XLim = [0 5.5];
% Fig1b.YLim = [0 1.6];
% Fig1b.Box = 'on';
% ylabel({'$L$'},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Times')
% xlabel({''},'FontUnits','points','Interpreter','latex','FontWeight','normal','FontSize',fontsize,'FontName','Times')

% set(gca,'LineWidth',1.5)


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
ylim([0 1.85])

print(1,'Fig5a','-depsc','-painters')
