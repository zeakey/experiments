function center = ResetCenter(center, sum_path_map, branches);

i = 2;
overbranchid = [];
while(1)
    for j = 1:length(branches)
        if(~isempty(find(overbranchid == j)))
            continue;
        end
        branch = branches{j};
        if(i > size(branch, 1))
            overbranchid = [overbranchid; j];
            continue;
        end
        cn = ConnectNum(branch(i,1),branch(i,2), sum_path_map);
        if(cn > 2)
            center = [branch(i,1),branch(i,2)];
            return;
        end
    end
    if(length(overbranchid) >= length(branches))
        break;
    end
    i = i + 1;
end


% function ij = IsJuncP(px,py, im_path_map)
% ij = 0;
% marks = [];
% for i = px-1:px+1
%     for j = py-1:py+1
%         if(im_path_map(i,j) > 0 & any([i,j] ~= [px, py]))
%             marks = [marks; im_path_map(i,j)];
%         end
%     end
% end
% if(length(marks) > 2)
%     if(any(marks) ~= marks(1))
%         ij = 1;
%     end
% end