function [branches] = GetBranchPath(bw, I0, endpoint)
%[branches] = GetBranchPath(bw, [x1,y1]);
% From center to each endpoint

[tm, tn] = size(endpoint);
[px, py] = find(bw ~= 0);
% [mx, my] = find(I0 ~= 0);
% dis = sqrt((mean(mx) - px).^2 + (mean(my) - py).^2);
[mx, my] = find(bw == max(max(bw)));
dis = sqrt((mx(1) - px).^2 + (my(1)  - py).^2);

l = find(dis == min(dis));
center = [px(l(1)), py(l(1))];
for i = 1 : tm;
    branches{i,1} = pathDFS1(bw, center, endpoint(i,:));
end

