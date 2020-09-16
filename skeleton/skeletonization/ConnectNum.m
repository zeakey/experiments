function cn = ConnectNum(x,y, skltn)
%计算（x，y）处的骨架连接数

mark = zeros(5, 5);
mark0 = zeros(5, 5);
x0 = x - 2;
y0 = y - 2;
if(skltn(x,y) ~= 0)
    for i = x-2:x+2
        for j = y-2:y+2
            if(skltn(i,j) ~= 0)
                mark0(i - x0 + 1, j - y0 + 1) = 1;
                if((i - x)^2 + (j - y)^2 >= 4)   
                    mark(i - x0 + 1, j - y0 + 1) = 1;
                end
            end
        end
    end
end
[L, num] = bwlabel(mark, 8);
[L0, num0] = bwlabel(mark0, 8);
cn = num - num0 + 1;