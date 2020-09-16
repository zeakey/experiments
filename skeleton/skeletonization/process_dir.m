clear all;
alpha = 9;
threshold = 0.001;

folder = 'C:\Users\zeake\Desktop\MSRA10K\gt_masks';
save_dir = 'C:\Users\zeake\Desktop\MSRA10K\skeleton_masks';

if ~isdir(folder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', folder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(folder, '*.png');
images = dir(filePattern);

for k = 1:length(images)
  fn = images(k).name;
  if exist(fullfile(save_dir, fn), 'file') == 2
      disp([fullfile(save_dir, fn), 'exists !']);
      continue;
  end
  fullfn = fullfile(folder, fn);
  im = imread(fullfn);
  [H, W] = size(im);
  bw1=im2bw(im);
  bw1 = imfill(bw1, 'holes');
  [m,n] = size(bw1);
  bwlb = bwlabel(bw1);
  if numel(unique(bwlb)) > 2
    skeleton = zeros(H, W);
    for i =2:size(unique(bwlb))
        try
              [skel_image1, skel_dist, I0, endpoint, branches, tendpoint,weights,ars,slrs,epl] = DSE(bwlb==i-1, 50, alpha,threshold);
        catch
              skel_image1 = ones(H, W);
        end
        skeleton1 = skel_image1==0;
        skeleton = skeleton + skeleton1;
    end
    skeleton = im2bw(skeleton);
  else
    [skel_image, skel_dist, I0, endpoint, branches, tendpoint,weights,ars,slrs,epl] = DSE(bw1,50, alpha,threshold);
    skeleton = skel_image==0;
  end
  
  imwrite(skeleton, fullfile(save_dir, fn))
  if mod(k, 20) == 0
      fprintf(1, 'Now reading %d of %d %s\n', k, length(images), fn);
  end
end