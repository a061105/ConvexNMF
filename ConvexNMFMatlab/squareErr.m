function [Z2,W2,err] = squareErr(Z, c, W, K0, R)

if length(c) == 0
	err = norm(R,'fro')^2;
	return;
end
K = length(c);

c_wnorm2 = zeros(K,1);
for k=1:K
				c_wnorm2(k) = c(k) * norm(W(k,:))^2;
end
[tmp,ind] = sort(c_wnorm2,'descend');
%[tmp,ind] = sort(c,'descend');

Z2 = Z(:,ind(1:min(K0,end)));

[N,K]=size(Z2);
W2 = inv(Z2'*Z2+1e-5*eye(K))*Z2'*R;

err = norm(R-Z2*W2,'fro')^2;
