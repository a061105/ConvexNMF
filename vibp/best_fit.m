% use to find the best fit Z from  learning to the true Z
function [fit_Z,hami]= best_fit (trueZ,rZ)
	%transpose first and then use pdist2 to mesure distance 
	[n,k]=size(trueZ);
	[~,ind]=min(pdist2(trueZ',rZ','hamming'),[],2);
	fit_Z = rZ(:,ind);	
	hami = sum(sum(abs(fit_Z-trueZ))/(n*k));
end
