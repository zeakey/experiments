function output = PathMap(bw, path)

[len, wid] = size(bw);
output = zeros(len, wid);
for i = 1:size(path,1)
    output(path(i,1), path(i,2)) = 1;
end