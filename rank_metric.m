function [hit,mrr,ndcg] = rank_metric(m, B, D, Test)
%RANK_METRIC 此处显示有关此函数的摘要
%   此处显示详细说明
[r,N]=size(B);
[r,M]=size(D);
B(B == -1) = 0;
D(D == -1) = 0;
[maxS,~] = max((Test(:)));
[minS,~] = min((Test(:)));
ndcg = zeros(m,1);
hit = zeros(m,1);
mrr = 0;
test_num = 0;
parfor u=1:N
    positive = find(Test(u,:) > 3.5);
    neg = find((Test(u,:) > 0)&(Test(u,:) < 4)); 
    mrr_u = zeros(m,length(positive));
    hit_u = zeros(m,length(positive));
    ndcg_u = zeros(m,length(positive));
    pred = (B(:,u)'*D);
    %pred = (1 - sum(B(:,u)&(~D),1)/r) * (maxS - minS) + minS;
    if (~isempty(positive))&&(~isempty(neg))
        if length(neg) > 1000
            rand('state',0);
            ffw=randperm(length(neg),1000);
            negative = neg(ffw);
        else
            negative = neg;
        end
        test_num = test_num + 1;
        for i = 1:length(positive)
            pos = positive(i);
            pred_list = [pred(pos),pred(negative)];
            % 随机排序
            randIndex = randperm(size(pred_list,2));
            positive_index = find(randIndex == 1);
            pred_list_new = pred_list(randIndex);
            [~,rank_index] = sort(pred_list_new,'descend');
            rank_ui = find(rank_index == positive_index);
            for k = 1:m
                if rank_ui <= k
                    hit_u(k,i) = 1;
                    mrr_u(k,i) = 1/rank_ui;
                    ndcg_u(k,i) = 1/log2(rank_ui + 1);
                end
            end
        end
        mrr = mrr + mean(mrr_u,2);
        ndcg = ndcg + mean(ndcg_u,2);
        hit = hit + mean(hit_u,2);
    end
end
mrr = mrr/test_num;
hit = hit/test_num;
ndcg = ndcg/test_num;
end

