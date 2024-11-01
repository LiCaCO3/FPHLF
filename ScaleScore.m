function s = ScaleScore(s,scale, maxS,minS)
%ScaleScore: scale the scores in user-item rating matrix to [-scale,
%+scale]. See footnote 2.
    s = (s-minS)/(maxS-minS); %S被放缩到了[0,1]之间
    s = 4*scale*s-3*scale; 
end