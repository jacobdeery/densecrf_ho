%% Plot speed up (old-time/new-time) by varying sigma, pixels and labels for each kernel

fpath = '/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/densecrf/old_new_ph/2007_000676/';

dim=[2,5];
sigma=[1,2,5,10,15,20];
imskip=[1, 2, 3, 4, 5];
labels=[2, 5, 10, 15, 21];

fname = [fpath '/timings_%u_%u_%u_%u.out'];

% speedup vs sigma
ims = 1;
l = 21;
epsname = [fpath '/timings_%u_%u_%u_sigma.eps'];
pdfname = [fpath '/timings_%u_%u_%u_sigma.pdf'];
for i = 1 : length(dim)
  data = cell(length(sigma), 1);
  speedup = zeros(length(sigma), 1);
  for j = 1 : length(sigma)
    data{j} = dlmread(sprintf(fname, dim(i), sigma(j), ims, l), '\t');
    data{j} = mean(data{j}, 1);
    speedup(j) = data{j}(3)/data{j}(4);
  end
  plot(sigma, speedup, '-ro', 'LineWidth',2,...
            'MarkerEdgeColor','r',...
            'MarkerFaceColor','r',...
            'MarkerSize',6);
  xlabel('Standard-deviation', 'FontSize', 22);
  ylabel('Speed-up', 'FontSize', 22);
  if (dim(i) == 2), title('Spatial kernel, d = 2', 'FontSize', 22); end;
  if (dim(i) == 5), title('Bilateral kernel, d = 5', 'FontSize', 22); end;
  print('-depsc2', sprintf(epsname, dim(i), ims, l));
  eps2pdf(sprintf(epsname, dim(i), ims, l), sprintf(pdfname, dim(i), ims, l));  
end

% speedup vs pixels
s = 1;
l = 21;
epsname = [fpath '/timings_%u_%u_%u_pixels.eps'];
pdfname = [fpath '/timings_%u_%u_%u_pixels.pdf'];
for i = 1 : length(dim)
  data = cell(length(imskip), 1);
  speedup = zeros(length(imskip), 1);
  for j = 1 : length(imskip)
    data{j} = dlmread(sprintf(fname, dim(i), s, imskip(j), l), '\t');
    data{j} = mean(data{j}, 1);
    speedup(j) = data{j}(3)/data{j}(4);
  end
  plot(imskip, speedup, '-ro', 'LineWidth',2,...
            'MarkerEdgeColor','r',...
            'MarkerFaceColor','r',...
            'MarkerSize',6);
  xlabel('No of pixels', 'FontSize', 22);
  ylabel('Speed-up', 'FontSize', 22);
  if (dim(i) == 2), title('Spatial kernel, d = 2', 'FontSize', 22); end;
  if (dim(i) == 5), title('Bilateral kernel, d = 5', 'FontSize', 22); end;
  print('-depsc2', sprintf(epsname, dim(i), s, l));
  eps2pdf(sprintf(epsname, dim(i), s, l), sprintf(pdfname, dim(i), s, l));  
end

% speedup vs labels
s = 1;
ims = 1;
epsname = [fpath '/timings_%u_%u_%u_labels.eps'];
pdfname = [fpath '/timings_%u_%u_%u_labels.pdf'];
for i = 1 : length(dim)
  data = cell(length(labels), 1);
  speedup = zeros(length(labels), 1);
  for j = 1 : length(labels)
    data{j} = dlmread(sprintf(fname, dim(i), s, ims, labels(j)), '\t');
    data{j} = mean(data{j}, 1);
    speedup(j) = data{j}(3)/data{j}(4);
  end
  plot(labels, speedup, '-ro', 'LineWidth',2,...
            'MarkerEdgeColor','r',...
            'MarkerFaceColor','r',...
            'MarkerSize',6);
  xlabel('No of labels', 'FontSize', 15);
  ylabel('Speed-up', 'FontSize', 15);
  if (dim(i) == 2), title('Spatial kernel, d = 2', 'FontSize', 15); end;
  if (dim(i) == 5), title('Bilateral kernel, d = 5', 'FontSize', 15); end;
  print('-depsc2', sprintf(epsname, dim(i), s, ims));
  eps2pdf(sprintf(epsname, dim(i), s, ims), sprintf(pdfname, dim(i), s, ims));  
end
