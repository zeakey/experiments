
function img2 = my_plot(img, list, dot_color, dot_size)

    if nargin == 3
        dot_size = 1;
    end
    
    if size(img, 3) == 3
        img2 = img;
    else
        if max(img(:)) == 1
            img = 255 * img;
        end
        img2 = zeros( size(img, 1), size(img, 2), 3 );
        img2(:, :, 1) = img;
        img2(:, :, 2) = img;
        img2(:, :, 3) = img;
        img2 = uint8(img2);
    end
    
    [rows, cols, nc] = size(img2);
    for n = 1 : size(list, 1)
        pnt = list(n, :);
        for r = max(1, pnt(1)-dot_size) : min(rows, pnt(1)+dot_size)
            for c = max(1, pnt(2)-dot_size) : min(cols, pnt(2)+dot_size)
                img2(r, c, :) = dot_color;
            end
        end
    end
    
    