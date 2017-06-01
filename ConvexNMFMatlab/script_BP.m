[A,Z_BP]=BPmeans(R,K);

[Z_true NaN(N,1) Z_BP]

tmp = R-Z_BP*A;
['loss=' num2str(sum(sum(tmp.*tmp)))]
