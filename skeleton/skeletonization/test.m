
alpha = 9.0;
threshold = 0.01;
dir = './'
sf = strcat(dir, 'dog-5.gif');


I = imread(sf);

I = I .* 255;
bw1=im2bw(I);


bw1 = imfill(bw1, 'holes');
[m,n] = size(bw1);




%skeleton pruning
tic;
[skel_image, skel_dist, I0, endpoint, branches, tendpoint,weights,ars,slrs,epl] = DSE(bw1,50, alpha,threshold);
toc;

%save the skeleton
skel_image = skel_image/2;

[sx,sy] = find(skel_image == 0);
list = [sx,sy];
skel_image = my_plot(skel_image, list, [0 0 0], 1);

bw1 = strcat(sf ,'-wskeleton.jpg');
imwrite(skel_image,bw1);
imshow(skel_image)



bw2 = strcat(sf ,'-orskeleton.jpg');
imwrite(skel_dist + I0,bw2);
% imshow(skel_dist)