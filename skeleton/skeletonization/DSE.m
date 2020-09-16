function [skel_image, skel_dist, I0, endpoint, branches, tendpoint,weights,ars,slrs,avgendpathlen] = DSE(bw, n_0, beta, threshold)
%skel_image:output skeleton
%skel_dist: distance transform
%I0:binary mask
%bw: input binary mask
%n_0: the number of vertices for the simified polygon by DCE, usually a constant between 30~50
%threshold: the stop threshold for DSE
%--------------------------------------------------------------------------------------------
%This version can not solve the binary shape with holes, which will be given in the forthcoming version.
%The authors of this software are Xiang Bai & Yao Xu, Huazhong Univ. of Science and Tech., Wuhan, China.
%Contact Email: {xiang.bai, quickxu}@gmail.com



[m,n]=size(bw);
% bw = imresize(bw,350/max(m,n));


bw1 = padarray(bw, [3,3]);
% Bs = bwboundaries(bw1);
% boundary = Bs{1};
%ShapeArea = size(find(bw1 > 0), 1);
I=1-bw1;
%%to show the original image
%figure;
%imshow(I,[]);

[bw,I0,x,y,x1,y1,aa,bb]=div_skeleton_new(4,1,1,I,n_0);
%[skel_dist,I0, endpoint] =SkeletonGrow(I,boundary);

% Ir = Reconstruct(bw);
% imshow(I0);
% hold on;
% [cx,cy] = find(Ir > 0);
% for i = 1:length(cx)
%     plot(cy(i), cx(i), '.r');
% end
skel_dist = bw;
orSKlen = size(find(skel_dist > 0), 1);
%orSKlen = DTlen(bw, bw);

I0= double(I0);
    
    [branches] = GetBranchPath(bw, I0, [x1,y1]);
%[branches] = GetBranchPath(skel_dist, I0, endpoint);    
endpathlens = [];
for i = 1:length(branches)
    branch = branches{i};
    endpathlens = [endpathlens;size(branch,1)];
end
avgendpathlen = mean(endpathlens);
endpoint = [x1,y1];

%tendpoint = endpoint;
%skel_image = orSKlen;
% threshold = 0.0005;%bone
%threshold = 0.005;

% [endpoint, branches, weight_r] = pruning_iteration(bw, I0, endpoint, branches, threshold);

[skel_image, endpoint, branches, tendpoint,weights,ars,slrs] = pruning_iteration1(skel_dist, I0, endpoint, branches, bw1, orSKlen, beta, threshold,avgendpathlen);

% by KAI ZHAO, remove padded pixels
skel_image = skel_image(4:end-3, 4:end-3);