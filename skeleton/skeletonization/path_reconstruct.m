function output = path_reconstruct(bw, path);
% reconstruct the original shape with skeleton
% bw denotes skeleton map

[len, wid] = size(bw);
output = zeros(len, wid);
 
%[m, n] = find(bw ~= 0);

for i = 1:size(path,1)
    radius = bw(path(i,1), path(i,2));
    output = recover_circle(path(i,1), path(i,2), round(radius), output);
end

function output = recover_circle(x0, y0, r0, bw);
%(x0, y0) denotes the coordinate of skeleton point
% r0 denotes the radius of (x0, y0);
% bw denotes the image to be reconstructed
[len, wid] = size(bw);
for i = max(1,x0-r0):min(x0+r0, len)
    for j = max(1, y0-r0):min(y0+r0, wid)
        if sqrt((i-x0)^2+(j-y0)^2)<=r0
           bw(i, j) = 1;
        end
    end
end

output = bw;