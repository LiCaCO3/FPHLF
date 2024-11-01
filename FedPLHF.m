function [B, D, rmse_min,mae_min,hit_max,mrr_max,ndcg_max] = FedPLHF(maxS, minS, S, ST, IDX, IDXT,Test, r, alpha, beta, dir, option)

[m,n] = size(S);
converge = false;
K = 10;
it = 1;

if isfield(option,'maxItr')
    maxItr = option.maxItr;
else
    maxItr = 50;
end
if isfield(option,'maxItr2')
    maxItr2 = option.maxItr2;
else
    maxItr2 = 5;
end
if isfield(option, 'Init')
   Init = option.Init;
else
   Init = True;
end
if Init
   if (isfield(option,'B0') &&  isfield(option,'D0') && isfield(option,'X0') && isfield(option,'Y0'))
       B0 = option.B0; D0 = option.D0;
   else
       U = rand(r, m);
       V = rand(r, n);
       B0 = sign(U); B0(B0 == 0) = 1;
       D0 = sign(V); D0(D0 == 0) = 1;
   end
else
    U = rand(r, m);
    V = rand(r, n);
    B0 = sign(U); B0(B0 == 0) = 1;
    D0 = sign(V); D0(D0 == 0) = 1;
end
if isfield(option,'debug')
    debug = option.debug;
else
    debug = false;
end

B = B0;
D = D0;
if debug
   [loss,obj] = FedPLHFobj(maxS,minS,S,IDX,B,D,alpha,beta);
   disp('Starting DCF...');
   
   disp(['loss value = ',num2str(loss)]);
   disp(['obj value = ',num2str(obj)]);
end

rmse_min = Inf;
mae_min = Inf;
ndcg_max = 0;
hit_max = 0;
mrr_max = 0;
delay = 0;

while ~converge
    fid=fopen([dir, '\train_result_lambda.txt'],'a');
    fprintf(fid,'binary_code_bit = %d; alpha = %f; beta = %f; K = %d; \n',r,alpha,beta,K);
    B0 = B;
    D0 = D;
    parfor i = 1:m
        %閫愪綅浼樺寲B鐭╅樀
        d = D(:,IDXT(:,i));
        b = B(:,i);
        b = local_update(b,d,ScaleScore(nonzeros(ST(:,i)),r,maxS,minS),alpha,maxItr2);
        B(:,i) = b;
    end
    parfor j = 1:n
        %閫愪綅浼樺寲D鐭╅樀
        b = B(:,IDX(:,j));
        d = D(:,j);
        d = global_update(d,b,ScaleScore(nonzeros(S(:,j)),r,maxS,minS),beta,maxItr2);
        D(:,j)=d;
    end

    if debug
        [loss,obj] = FedPLHFobj(maxS,minS,S,IDX,B,D,alpha,beta);
        disp(['loss value = ',num2str(loss)]);
        disp(['obj value = ',num2str(obj)]);
    end
    disp(['DCF at bit ',int2str(r),' Iteration:',int2str(it)]);
    
    [rmse,mae] = rating_loss(Test, B', D');
    [hit_sp,mrr_sp,ndcg] = rank_metric(10, B, D, Test);
    
    fprintf('round %d : RMSE = %f MAE = %f NDCG = %f HIT = %f MRR = %f\n',it,rmse,mae,ndcg(10),hit_sp(10),mrr_sp(10));
    fprintf(fid,'round %d : RMSE = %f MAE = %f NDCG = %f HIT = %f MRR = %f\n',it,rmse,mae,ndcg(10),hit_sp(10),mrr_sp(10));
    fclose(fid);
    if (mrr_sp(10) > mrr_max)
        mrr_max = mrr_sp(10);
        delay = 0;
    else
        delay = delay + 1;
    end
    if (ndcg(10) > ndcg_max)
        ndcg_max = ndcg(10);
        delay = 0;
    else
        delay = delay + 1;
    end
    
    if (hit_sp(10) > hit_max)
        hit_max = hit_sp(10);
        delay = 0;
    else
        delay = delay + 1;
    end
    if (mae < mae_min)
        mae_min = mae;
        delay = 0;
    else
        delay = delay + 1;
    end
    if (rmse < rmse_min)
        rmse_min = rmse;
        delay = 0;
    else
        delay = delay + 1;
    end
    
    if delay > 30
        break;
    end

    if it >= maxItr || (sum(sum(B~=B0)) == 0 && sum(sum(D~=D0)) == 0)
        converge = true;
    end
    it = it+1;
end

end

function [loss,obj] = FedPLHFobj(maxS,minS,S,IDX,B,D,alpha,beta)
[~,n] = size(S); %size锛堬級鑾峰彇鐭╅樀鐨勮鏁板拰鍒楁暟 鍒楁暟涓簄
r = size(B,1);
loss = zeros(1,n);
B(B == -1) = 0;
D(D == -1) = 0;
% 这里为了加快训练速度，给出上传聚合后的计算结果
parfor j = 1:n
    dj = D(:,j); %公开实体二进制表征 中心服务器分发给每个客户端
    Bj = B(:,IDX(:,j));  %私有实体二进制表征 
    pred = (1 - sum(Bj&(~dj),1)/r)*4*r - 3*r;
    Sj = ScaleScore(nonzeros(S(:,j)),r,maxS,minS);
    loss(j) = sum((Sj' - pred).^2);
end
loss = sum(loss); %杩斿洖姣忎釜鐢ㄦ埛璇勫垎椤圭殑鎹熷け
B(B == 0) = -1;
D(D == 0) = -1;
% obj = loss - 2*alpha*trace(B*X')- 2*beta*trace(D*Y');
obj = loss + alpha*sum(B,'all') + beta*sum(D,'all');
end

function b = local_update(b,d,S,alpha,maxItr)
r = size(b,1);
no_update_count = 0;
for it = 1:maxItr
    for k = 1:r
        db = d'*b;
        db_notk = db - d(k,:)'*b(k);
        b_notk_sum = sum(b) - b(k);
        d_notk_sum = (sum(d,1) - d(k,:))';
        detla = sum((S - (db_notk - b_notk_sum + d_notk_sum - 1)).*(d(k,:) - 1)');
        cons = alpha * b_notk_sum;
        bk_new = -(detla - cons);
        if bk_new > 0
            if b(k) < 0
                no_update_count = no_update_count + 1; 
            else
                b(k) = -1;
            end
        else
            if b(k) > 0
                no_update_count = no_update_count + 1; 
            else
                b(k) = 1;
            end
        end
    end
    if (no_update_count==r)
        break;
    end
end
end

function d = global_update(d,b,S,beta,maxItr)
r = size(d,1);
no_update_count = 0;
for it = 1:maxItr
    for k = 1:r
        bd = b'*d;
        bd_notk = bd - b(k,:)'*d(k);
        b_notk_sum = (sum(b,1) - b(k,:))';
        d_notk_sum = sum(d) - d(k);
        detla = (S - (bd_notk - b_notk_sum + d_notk_sum - 1)).*(b(k,:) + 1)'; % 每个本地上传的梯度组成的向量
        cons = beta * d_notk_sum;
        dk_new = -sum(sign(detla - cons)); % 聚合上传 sign(detla - cons)
        if dk_new > 0
            if d(k) < 0
                no_update_count = no_update_count + 1; 
            else
                d(k) = -1;
            end
        else
            if d(k) > 0
                no_update_count = no_update_count + 1; 
            else
                d(k) = 1;
            end
        end
    end
    if (no_update_count==r)
        break;
    end
end
end
