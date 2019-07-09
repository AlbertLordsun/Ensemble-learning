function  [subsetx, subsety, subsetw] = SubsetSearch(label, feature, w)

selected_idx = [];
% select 20 for each subset
cnt = zeros(10);        %label class
[~, t2] = sort(w, 'descend');
for i = 1 : size(label, 1);
    idx = t2(i);
    if (cnt(label(idx)) >= 20)
        continue;
    end
    cnt(label(idx)) = cnt(label(idx)) + 1;
    selected_idx = [selected_idx idx]; %#ok<*AGROW>
end

moreselect = floor(numel(label) * rand(1));
[t1, ~] = sort(w, 'descend');
t1 = ceil(t1/sum(t1) * 20);
temp = 1:size(label,1);
for i = 1 : size(label,1)
    temp = [temp i*ones(1, t1(i))];
end
idx = randperm(size(temp,2));
selected_idx = [selected_idx temp(idx(1:moreselect))];

selected_idx = unique(selected_idx);
subsetx = feature(selected_idx, :);
subsety = label(selected_idx, :);
subsetw = w(selected_idx, :);
end