function [lambda,mu_k,out] = fast_tensor4_whitening(m1,M1,lambda_list,theta_lst,data,vari,vari4,stop,u,W,K)
L = 10;
N = 10;
truncated = 0;
[d num] = size(data);
theta_list = zeros(K,L);
lambda= zeros(1,L);
lambda_len = length(lambda_list);
if truncated == 1
    [sorted index] = sort(abs(M1));
    truncated_index = index(1:round(d*0.3));
end

for i = 1:L
    mu_k = randn(K,1);
    v = W*(mu_k./norm(mu_k));
    M_temp = [];
    for j = 1:N
        data_v = data'*v;
        cross = data*(data_v)./num;
        M1_v = M1'*v;
        M2 = cross - M1.*(M1_v)- vari.*v;
   
        cross1 = data*((data_v).^2)./num;
        M3 = cross1 -M1.*(v'*M2)- M2.*(M1_v)-M2.*(M1_v)-M1.*((M1_v)^2);
        M3 = M3 - m1.*(sum(v.^2)) - v.*(m1'*v).*2;
        %M2_v = M2'*v;
        %cross2 = data*((data_v).^3)./num;
        %M4 = cross2 - M1.*((M1_v)^3) - 3.*M2.*((M1_v)^2) - 3.*M1.*(M2_v).*(M1_v);
        %M4 = M4 - 3.*M2.*(M2_v);
        %M4 = M4 - 3.*M3.*(M1_v) - M1.*(M3'*v);
        %m2 = data*(data'*v.*((u*(data)).^2)')./num;
        if u == 0
            m2 = vari.*(cross - vari.*v) + vari4.*v;
        else
            m2 = data*(data'*v.*((u*(data)).^2)')./num;
        end
        %M4 = M4 - 3.*m2.*(sum(v.^2)) - 3.*(m2'*v).*v ;   
        %vari4 = mean((u*(data - repmat(M1,[1,num]))).^4);
        %M4 = M4 + 5.*vari4.*(v.^3);
        wM3vv = W'*M3;
        if lambda_len ~= 0
            wM3vv = wM3vv - (theta_lst(:,stop:end))*((theta_lst(:,stop:end)'*mu_k).^2.*(lambda_list(:,stop:end)'));
        end

        if j < N
            mu_k = wM3vv; 
            mu_k = mu_k./norm(mu_k);
            v = W*mu_k;
        end 
        %out = [mm2 m2];
    end
    theta_list(:,i) = mu_k;
    lambda(i)=mu_k'*wM3vv;
    
end
out = [lambda' theta_list'];
[maxi, index] = max(lambda);
mu_k = theta_list(:,index);
v = W*mu_k;

N = 50;
for j = 1:N
        data_v = data'*v;
        cross = data*(data_v)./num;
        M1_v = M1'*v;
        M2 = cross - M1.*(M1_v)- vari.*v;
   
        cross1 = data*((data_v).^2)./num;
        M3 = cross1 -M1.*(v'*M2)- M2.*(M1_v)-M2.*(M1_v)-M1.*((M1_v)^2);
        M3 = M3 - m1.*(sum(v.^2)) - v.*(m1'*v).*2;
        M2_v = M2'*v;
        %cross2 = data*((data_v).^3)./num;
        %M4 = cross2 - M1.*((M1_v)^3) - 3.*M2.*((M1_v)^2) - 3.*M1.*(M2_v).*(M1_v);
        %M4 = M4 - 3.*M2.*(M2_v);
        %M4 = M4 - 3.*M3.*(M1_v) - M1.*(M3'*v);
%         if u == 0
%             m2 = vari.*(cross - vari.*v) + vari4.*v;
%         else
%             m2 = data*(data'*v.*((u*(data)).^2)')./num;
%         end
%         M4 = M4 - 3.*m2.*(sum(v.^2)) - 3.*(m2'*v).*v ;   
%         %vari4 = mean((u*(data - repmat(M1,[1,num]))).^4);
%         M4 = M4 + 5.*vari4.*(v.^3);
        wM3vv = W'*M3;
        if lambda_len ~= 0
            wM3vv = wM3vv - (theta_lst(:,stop:end))*((theta_lst(:,stop:end)'*mu_k).^2.*(lambda_list(:,stop:end)'));
        end
%         if truncated == 1
%             M4(truncated_index) = 0;
%         end
        if j < N
            mu_k = wM3vv; 
            mu_k = mu_k./norm(mu_k);
            v = W*mu_k;
        end 

end
%out = [M3 M4];
lambda = mu_k'*wM3vv;
