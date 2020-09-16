function bw1=preprocess(bw);
[m,n] = size(bw);
% if max([m,n])>350,%350
%     [bw]=imresize(bw,300/max([m,n]));
% end

if max(max(bw))>1
    bw=im2bw(bw);
end

[m,n] = size(bw);
for i=1:m
    bw(i,1)=0;
    bw(i,2)=0;
    bw(i,n)=0;
    bw(i,n-1)=0;
end

for j=1:n;
    bw(1,j)=0;
    bw(2,j)=0;
    bw(m,j)=0;
    bw(m-1,j)=0;
end

bw1=bw;


