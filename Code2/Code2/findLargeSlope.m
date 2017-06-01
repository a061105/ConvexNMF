function [indexi] = findLargeSlope(D,setIndexi)

diagD = D;

diagD_tmp = diagD;
diagD_tmp(diagD_tmp < 0.00001) = 0.001;
rateD = diagD(1:end-1)./diagD_tmp(2:end);
[maxi, indexi] = max(rateD(2:end));
indexi = indexi +1;
if setIndexi ~=0
    indexi = setIndexi;
end
