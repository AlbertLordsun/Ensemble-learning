function [py, vote_vl] = AdaBoostClassify_Debug(xtest, H, encode_classifer, encode_rule, w_c, T, vote_vl)

d_y = 10;  %label class
d_s = size(xtest, 1);
py =zeros(d_s, 1);
t = T;

[~, B] = AdaBoostWeakLearnerClassify(xtest, H(t).hypothesis, encode_classifer, encode_rule, w_c);
for j = 1 : d_s
    for k = 1 : d_y
        vote_vl(j,k) = vote_vl(j,k) + log(1/H(t).beta)*B(j,k);
    end
end


for i = 1 : d_s
    [~, py(i)] = max(vote_vl(i, :)) ;
end
