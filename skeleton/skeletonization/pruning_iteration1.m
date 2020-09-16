function [skel_image, endpoint, branches, tendpoint,weights,ars,slrs] = pruning_iteration1(bw, I0, endpoint, branches, ShapeIm, orSKlen, beta, threshold,avgendpathlen)
%----------------------------------------------------------------------- 
%Desc: iterative  algorithm  to  prune  the  skeleton. 
% Para:bw denotes skeleton map
%      In bw, Non-zero point present the radius of its maximal disk
%      I0 the object image, m ¡Á n matrix ,Non-zero point present the
%      position of the shape
%      endpoint, branches, threshold its' own name present itself
%Return: skel_image,the skeleton image
%      endpoint, branches its' own name present itself  
%      tendpoint is a temp to endpoint
%----------------------------------------------------------------------- 
len = length(branches);
[tm, tn] = size(bw);
sum_im = zeros(tm, tn);
sum_path_map = zeros(tm, tn);
for i = 1:len
    im_path_re{i,1} = path_reconstruct(bw, branches{i,1});
    im_path_map{i,1} = PathMap(bw, branches{i,1});
    sum_im = sum_im + im_path_re{i,1};
    sum_path_map = sum_path_map + im_path_map{i,1};
end

 
%weight_r = 0;
ProS = 0;
pre_ProS = 0;
%compute the center point
[px, py] = find(bw ~= 0);
[mx, my] = find(bw == max(max(bw)));
dis = sqrt((mx(1) - px).^2 + (my(1)  - py).^2);
l = find(dis == min(dis));
center = [px(l(1)), py(l(1))];
weights = [];
ars = [];
slrs = [];
%alphas = [];
%alpha = stdDT(sum_path_map, bw)^4 * 6.6;
ProS_m = 0;
while (1)
    if(size(endpoint,1) < 3)
        break;
    end
    [weight, ar, slr] = optimalSKfuc2(sum_path_map,sum_im, ShapeIm, orSKlen, beta,avgendpathlen);
    weights = [weights;weight];
    ars = [ars;ar];
    slrs = [slrs;slr];
    [tendpoint] = endpoint;
    %alphas = [alphas;stdDT(sum_path_map, bw)];
    %alpha = stdDT(sum_path_map, bw)^2 * 4.3;
    %alpha = mean(alphas)^2 * 4.9;
    [ProS,endpoint, branches, im_path_re, sum_im, im_path_map, sum_path_map, remove_endpoint, remove_branch]=remove_one(endpoint, branches, im_path_re, sum_im,im_path_map, sum_path_map,bw,center, ShapeIm, orSKlen, beta,avgendpathlen);
    
    %check the structure of skeleton
    cx = center(1);
    cy = center(2);
    cn = ConnectNum(cx,cy, sum_path_map);
    marks = [];
    for i = cx-1:cx+1
        for j = cy-1:cy+1
            if(sum_path_map(i,j) > 0 & any([i,j] ~= [cx, cy]))
                marks = [marks; sum_path_map(i,j)];
            end
        end
    end
   if(~isempty(find(marks == 1)) & cn < 3)
      fprintf(1,'need reset center\n');
      sum_path_map = zeros(tm, tn);
      center = ResetCenter(center, sum_path_map, branches);
      [branches] = GetNewBranchPath(bw, center, endpoint);
      len = length(branches);
      for i = 1:len
          im_path_re{i,1} = path_reconstruct(bw, branches{i,1});
          im_path_map{i,1} = PathMap(bw, branches{i,1});
          sum_path_map = sum_path_map + im_path_map{i,1};
      end
   end
   if(ProS >= ProS_m)
        ProS_m = ProS;
        endpoint_f = endpoint;
        branches_f = branches;
        im_path_re_f = im_path_re;
        im_path_map_f = im_path_map;
        sum_im_f = sum_im;
        sum_path_map_f = sum_path_map;
    end
end
ShapeArea = length(find(ShapeIm>0));

endpoint = endpoint_f;
branches = branches_f;
End_num = length(branches);
im_path_re = im_path_re_f;
im_path_map = im_path_map_f;
sum_im = sum_im_f;
sum_path_map = sum_path_map_f;
checking = 0;
while(checking&End_num)
     End_lengths = [];
     End_areas = [];
    for i = 1:End_num
        End_length = length(find(sum_path_map) > 0) - length(find(sum_path_map - im_path_map{i}) > 0);
        End_area = length(find(sum_im) > 0) - length(find(sum_im - im_path_re{i}) > 0);
        End_lengths = [End_lengths;End_length];
        End_areas = [End_areas;End_area];
    end
    [p, s] = sort(End_areas);
    for i = 1:length(p)
        if(p(i) >= ShapeArea * 0.01)
            checking = 0;
            break;
        else
            no = s(i);
            if(End_lengths(no) <  orSKlen * 0.025)
                sum_path_map = sum_path_map - im_path_map{no};
                sum_im = sum_im - im_path_re{no};
                if no == 1
                   [endpoint] = [endpoint(2:End_num,:)];
                   [branches] = {branches{2:End_num,1}}';
                   [im_path_re] = {im_path_re{2:End_num,1}}';
                   [im_path_map] = {im_path_map{2:End_num,1}}';
                   flag = 1;
                elseif no == End_num
                   [endpoint] = [endpoint(1:End_num-1,:)];
                   [branches] = {branches{1:End_num-1,1}}';
                   [im_path_re] = {im_path_re{1:End_num-1,1}}';
                   [im_path_map] = {im_path_map{1:End_num-1,1}}';
                   flag = 2;
                else
                   [endpoint] = [endpoint(1:no-1,:);endpoint(no+1:End_num, :)];
                   [branches] = {branches{1:no-1,1},branches{no+1:End_num,1}}';
                   [im_path_re] = {im_path_re{1:no-1,1},im_path_re{no+1:End_num,1}}';
                   [im_path_map] = {im_path_map{1:no-1,1},im_path_map{no+1:End_num,1}}';
                   flag = 3;
                end
                fprintf(1, 'checking\n');
%                  checking = 0;
                End_num =End_num - 1;
                break;
            end
        end
    end
end

temp2 = ones(tm, tn);
[tm1, tn1] = size(endpoint);
for i = 1:tm1
    [temp1] = branches{i ,1};
    [tm2, tn2] = size(temp1);
    for j = 1:tm2
        temp2(temp1(j, 1), temp1(j, 2)) = 0;
    end
end

%temp2 = imresize(temp2,350/max(tm,tn));
% [temp1] = remove_branch;
% [tm2, tn2] = size(temp1);
% for j = 1:tm2
%     temp2(temp1(j, 1), temp1(j, 2)) = 0;
% end

%I0 = imresize(I0, 350/max(tm,tn));
skel_image = temp2+I0;



function  [ProS_m, endpoint, branches, im_path_re, sum_im, im_path_map, sum_path_map, remove_endpoint, remove_branch] = remove_one(endpoint, branches, im_path_re, sum_im,im_path_map, sum_path_map,bw,center, ShapeIm, orSKlen, beta, avgendpathlen)
%----------------------------------------------------------------------- 
%Desc: remove one end branch with the lowest weight 
%----------------------------------------------------------------------

len = length(branches);
[bwm,bwn] = size(bw);

%tlength = length(find(sum_im~=0));
ProS = [];
for i = 1:len
    SKareaIm = sum_im - im_path_re{i,1};
    %EndArea = size(find(sum_im > 0), 1) - size(find(SKareaIm > 0), 1);
    SKIm = sum_path_map - im_path_map{i,1};
    %EndLength = size(find(sum_path_map > 0), 1) - size(find(SKIm > 0), 1);
    ProS(i) = optimalSKfuc1(SKIm,SKareaIm, ShapeIm, orSKlen, beta,avgendpathlen);
end
[p, s] = sort(ProS, 'descend');
ProS_m = p(1);
no = s(1);

% tendpoint = endpoint;
% tbranches = branches;
% tim_path_re = im_path_re;
% tsum_im = sum_im;
remove_endpoint = endpoint(no,:);
remove_branch = branches{no,1};
sum_im = sum_im - im_path_re{no,1};
sum_path_map = sum_path_map - im_path_map{no,1};


if no == 1
   [endpoint] = [endpoint(2:len,:)];
   [branches] = {branches{2:len,1}}';
   [im_path_re] = {im_path_re{2:len,1}}';
   [im_path_map] = {im_path_map{2:len,1}}';
   flag = 1;
elseif no == len
   [endpoint] = [endpoint(1:len-1,:)];
   [branches] = {branches{1:len-1,1}}';
   [im_path_re] = {im_path_re{1:len-1,1}}';
   [im_path_map] = {im_path_map{1:len-1,1}}';
   flag = 2;
else
   [endpoint] = [endpoint(1:no-1,:);endpoint(no+1:len, :)];
   [branches] = {branches{1:no-1,1},branches{no+1:len,1}}';
   [im_path_re] = {im_path_re{1:no-1,1},im_path_re{no+1:len,1}}';
   [im_path_map] = {im_path_map{1:no-1,1},im_path_map{no+1:len,1}}';
   flag = 3;
end


% return;
%compute numbers of main branches from center point
% [path_num] = main_path(branches);
% if path_num == 1
%     [junctionPoints] = findNearJunction([bwm,bwn],branches,center); 
%     if (isempty(junctionPoints))
%         [endpoint,branches,im_path_re,sum_im] = all_recovery(tendpoint,tbranches,tim_path_re,tsum_im);
%         run = 0;   
%     elseif (~isempty(junctionPoints))
%        
%         center = junctionPoints(1,:);
%         [endpoint,sum_im] = endpoint_recovery(tendpoint,tsum_im);
%         [branches] = GetNewBranchPath(bw, center, endpoint);
%         len = length(branches);
%         for i = 1:len
%             im_path_re{i,1} = path_reconstruct(bw, branches{i,1});
%         end
%           
%     end
% end





function  [endpoint,sum_im] = endpoint_recovery(tendpoint,tsum_im)
%recover the endpoint and sum_im 
endpoint = tendpoint;
sum_im = tsum_im;

function [endpoint,branches,im_path_re,sum_im] = all_recovery(tendpoint,tbranches,tim_path_re,tsum_im)
%recover endpoint,branches,im_path_re,sum_im
endpoint = tendpoint;
branches = tbranches;
im_path_re = tim_path_re;
sum_im = tsum_im;

function [branches] = GetNewBranchPath(bw, center, endpoint)
%recompute branches path again
[tm, tn] = size(endpoint);
for i = 1 : tm;  
    branches{i,1} = pathDFS1(bw, center, endpoint(i,:));
end

function [path_num] = main_path(branches)
%compute numbers of main branches from center point
branchtemp = branches{1,1};
center_path_point(1,:) = branchtemp(10,:);
for i = 2:length(branches)
     branchtemp = branches{i,1};
     tr = 0;
     [mm,nn] = size(center_path_point);
     for j = 1:mm
        
         if branchtemp(10,:) == center_path_point(j,:)
            tr = 1;
            break;
         end
        
     end
     if tr == 0
        [center_path_point] = [center_path_point;branchtemp(10,:)];
     end    
end
[mm,nn] = size(center_path_point);
path_num = mm;