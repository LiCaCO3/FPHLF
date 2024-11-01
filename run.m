%all_dataset = {'Ciao','Epinion','Hetrec-ML','Ml1M','Yelp','Ml25M'}; %dataset_name = dataset_name{1};
all_dataset = {'Epinion'}; %dataset_name = dataset_name{1};
for dataset = all_dataset
    dataset_name = dataset{1};
    dir =['D:\Lishihui\EXPERIMENTS\FPHLF_CSCWD\', dataset_name];
    dataset_path = ['D:\Lishihui\BDELFA\new_data2\',dataset_name];
    load([dataset_path,'\test.mat']);
    load([dataset_path,'\train.mat']);

    %target bit size
    binary_bit = [64];
    K = 10;
    paraments = [0.0001];
    option.debug = true;

    %number of iterations
    option.maxItr = 100;
    option.maxItr2 = 5;

    S = train;
    Test = test;
    ST = S';
    IDX = (S~=0);
    IDXT = IDX';
    [rows, cols] = size(S);
    [maxS,~] = max((S(:)));
    [minS,~] = min((S(:)));
   
    for r = binary_bit
        for alpha = paraments
            for beta = paraments
                %apply initialization
                option.Init = false;
                option.B0 = randi([0,1], r, rows) * 2 - 1;
                option.D0 = randi([0,1], r, cols) * 2 - 1;
                option.X0 = rand(r, rows) * 2 - 1;
                option.Y0 = rand(r, cols) * 2 - 1;
                
                rmse_min = Inf;
                mae_min = Inf;
                ndcg_max = 0;
                hit_max = 0;
                mrr_max = 0;
                [B,D,rmse,mae,hit,mrr,ndcg] = FedPLHF(maxS,minS,S, ST, IDX, IDXT,Test, r, alpha, beta, dir, option);
                if (mrr > mrr_max)
                    mrr_max = mrr;
                end
                if (ndcg > ndcg_max)
                    ndcg_max = ndcg;
                end
                if (hit > hit_max)
                    hit_max = hit;
                end
                if (mae < mae_min)
                    mae_min = mae;
                end
                if (rmse < rmse_min)
                    rmse_min = rmse;
                end
                fid=fopen([dir,'\rating_result_lambda.txt'],'a');
                fprintf(fid,'binary_code_bit = %d; alpha = %f; beta = %f; K = %d; \n',r,alpha,beta,K);
                fprintf(fid,'RMSE = %f \t MAE = %f \t  Hit = %f \t MRR = %f \t NDCG = %f\n',rmse_min,mae_min,hit_max,mrr_max,ndcg_max);
                fclose(fid);
            end
        end

    end
end
