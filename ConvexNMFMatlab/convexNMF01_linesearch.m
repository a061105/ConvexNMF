%%Solving the convex NMF problem:
%%
%%   min_{Z\in{0,1}^{N,K}, V>=0}  \frac{1}{2}\|R-ZV'\|_F^2 + \frac{1}{2}\|V\|_F^2 + Lambda*\|ZZ'\|_{S}
%%
%%by solving:
%%
%%   min_{M} max_{A}  \frac{-1}{2} tr( (G+A)*(G+A)' M ) + tr(R'*A) - \frac{1}{2}\|A\|_F^2 + Lambda*\|M\|_S
%%
%%where G= Z' \ beta,   B = [-Z'A]_+  (s.t. B+Z'A=[Z'A]_+)

function [c,Z,V,A] = convexNMF01( R, Lambda, Tau, Z0 )

TOL = 1e-8;
T = 1000;
T2 = 1;
%T_A = 100;
SDP_iter = 100;
SDP_rank = 1;

[N,K0] = size(Z0);
[N,D] = size(R);
Z = [];
c = [];

tol_rate = 0.01;
f = @(Z1,z1,thd) sum( Z1~=(z1*ones(1,size(Z1,2))) ) <= thd;

last_obj = 1e300;
eta_rate = 1e-4;
avg_trial=0;
for t = 1:T
	
	if length(c)~=0
		Zc = Z*diag(sqrt(c));
	else
		Zc = zeros(N,1);
	end
	
	%find A*
	%[V,A,loss] = CDS_D( R, Zc, T_A, 1);
	[V,A] = LS_solve( R, Zc, Tau );
	%dump info
	%%%!!!PrimalLoss forgot to Multplier Tau !!
	dobj = NMFDualLoss( R, Zc, A, Tau ) + Lambda*sum(c);
	pobj = NMFPrimalLoss( R, Zc, V, Tau ) + Lambda*sum(c);
	err = squareErr(Z,c,K0,R);
	['t=' num2str(t) ', d-obj=' num2str(dobj) ', p-obj=' num2str(pobj) ', nnz(c)=' num2str(nnz(c)) ', err=' num2str(err) ', #trial=' num2str(avg_trial)]
	%['t=' num2str(t) ', nnz(c)=' num2str(nnz(c))]
	if dobj > last_obj + 1e-1
		'warning: eta_rate decreased due to increased obj'
		%eta_rate = eta_rate / 10;
		break;
	end
	last_obj = dobj;

	if( t==T )
		V = V';
		break;
	end

	%compute gradient
	%grad_M = gradient_M( Zc, A );
	grad_M = gradient_M( A, Tau );
	
	%find greedy direction & add to active set
	%[z, obj] = boolIQP(-grad_M*2);
	%[z, obj] = boolIQP(-grad_M*2-0.1*eye(N));
	%if t > 70 
	%	SDP_iter = 1000;
	%	T2 = 100;
	%end
	z = MixMaxCut(A,SDP_rank,SDP_iter);
	'maxcut done'

	if ~inside( Z, z )
		Z = [Z z];
		c = [c;0.0];
	end
	
	%fully corrective by prox-GD
	k = length(c);
	h = diag_hessian(Z);
	
	%eta = eta_rate/(max(h)*k)*Tau;
	eta0 = eta_rate/(max(h))*Tau;
	%{
	for t2 = 1:T2*k

					grad_c = gradient_c(Z,c,A, Tau);
					c = prox( c - eta*grad_c, eta*Lambda );

					%find A*
					Zc = Z*diag(sqrt(c));
					%[V,A,loss] = CDS_D( R, Zc, T_A, 1);
					[V,A] = LS_solve( R, Zc, Tau );
	end
	%}

	ZTR = Z'*R;
	ZTZ = Z'*Z;
	avg_trial = 0;
	for t2 = 1:T2
			[W,grad_c] = LS_solve2(ZTR, ZTZ, c, Tau);
			
			eta = eta0;
			%c = prox( c - eta*grad_c, eta*Lambda );
			
			for trial = 1:10 %line search
						%prox update
						c2 = prox( c - eta*grad_c, eta*Lambda );
						
						%Greedy update
						%c_tmp = prox( c - eta*grad_c, eta*Lambda );
						%[val, ind] = sort(abs(c_tmp-c),'descend');
						%c2 = c;
						%c2(ind(1:min(5,end))) = c_tmp(ind(1:min(5,end)));
						%find new V and test func
						[W,grad_c2] = LS_solve2(ZTR, ZTZ, c2, Tau);
						pobj = norm(R-Z*W,'fro')^2/2 + Tau*trace(W'*diag(1/c2)*W)/2 + Lambda*sum(c2);
						if pobj < last_obj
								break;
						else
						  	eta = eta/10;
						end
			end
			avg_trial = avg_trial + trial;
			
			c = c2;
			last_obj = pobj;
			
	end
	%avg_trial = avg_trial / T2;

	%shrink c and Z for j:cj=0
	Z = Z(:,c'>TOL);
	c = c(c>TOL);
	
	'prox-GD done'

	P = [c'];
	for k = 1:size(Z0,2)
		P = [f(Z,Z0(:,k),N*tol_rate);P];
	end
	P
	%Z(:,1:min(10,end))
	%save 'Z_bash4' Z -ascii;
	%save 'c_bash4' c -ascii;
end


end

function is_inside = inside( Z, z )
	
	is_inside = 0;
	for i = 1:size(Z,2)
		if  all(Z(:,i) == z)
			 is_inside = 1;
		end
	end
end

function M = compute_M(c,Z)
	
	[n,k] = size(Z);
	M = zeros(n,n);
	for j = 1:k
		M = M + c(j)*Z(:,j)*Z(:,j)';
	end
end

% here A must be a maximizer of dual loss
function grad_M = gradient_M(A, Tau)
	
	%B = max(-Zc'*A,0);
	%G = Zc'\B;
	
	%grad_M = -(G+A)*(G+A)'/2;
	grad_M = -A*A'/2/Tau;
end

function grad_c = gradient_c(Z,c,A, Tau)
	
	k = length(c);
	grad_c = zeros(k,1);
	%Zc = Z*diag(sqrt(c));
	grad_M = gradient_M(A, Tau);
	for j=1:k
		grad_c(j) = Z(:,j)'*grad_M*Z(:,j);
	end
end

function h = diag_hessian(Z)
	k = size(Z,2);
	h = zeros(k,1);
	for i = 1:k
		h(i) = (Z(:,i)'*Z(:,i)).^2;
	end
end

