function [rmse,mae] = rating_loss(R, P, Q)
[I,J,V] = find(R);
[~,r] = size(P);
[maxS,~] = max((R(:)));
[minS,~] = min((R(:)));
P(P == -1) = 0;
Q(Q == -1) = 0;
pred_val = sum(P(I,:)&(~Q(J,:)),2);
pred_temp = 1 - pred_val/r;
pred = pred_temp * (maxS - minS) + minS;
rmse = sqrt( mean((V - pred).^2) );
mae = mean( abs((V - pred)) );
end

