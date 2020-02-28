files = {'RD_0.tif','RD_20.tif','RD_40.tif','RD_80.tif','RD_120.tif','RD_160.tif'}
names = {'RD_0_crop.tif','RD_20_crop.tif','RD_40_crop.tif','RD_80_crop.tif','RD_120_crop.tif','RD_160_crop.tif'}

for i=1:length(files)
  f = files{i};
  g = names{i};
  I = imread(f);
  rect = [150,500,1300,1050];
  J = imcrop(I(:,:,1:3),rect);
  imwrite(J,g);
end
