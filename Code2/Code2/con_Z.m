% use to  reconstruct the matrix D = A Z where Z is a binary matrix
function [re_error,Z] = con_Z(D,A)    
    [n,m]=size(D);   
	% D = AZ
	[~,k]=size(A);
	Z=zeros(k,m);
    for i=1:m
     % go through possible combination
      c_error=inf;
	     for j=0:k
		      i_temp=nchoosek(1:1:k,j);
		      [w,~]=size(i_temp);
		      for t=1:w 
		      	temp=zeros(1,k);
		      	temp(i_temp(t,:))=1;
		      	loc_error=norm(D(:,i)-A*(temp'),'fro')^2;
		      	if (loc_error < c_error)
 					c_error=loc_error;
 					Z(:,i)=temp';	
 			  	end
		      end

	     end
    end
    re_error = norm(D-A*Z,'fro')^2;
end
