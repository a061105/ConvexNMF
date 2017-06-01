rand('seed',1);
randn('seed',1);
%synthetic data
NOISE_LEVEL = 1e-1;
LAMBDA =0.1;
n=20;
k=4;

z1 = [1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; 
z2 = [0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0]; 
z3=  [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0]; 
z4=  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1]; 
Z_true=[z1;z2;z3;z4]; 

M_true = Z_true'*Z_true;
R= M_true + NOISE_LEVEL*randn(n); 
R=(R+R')/2

%solve
[c,Z] = boolLasso(R,LAMBDA); 

%find top support
[c2,ind]=sort(c,'descend'); 
Z2 = Z(:,ind);

c = c2'
Z = Z2(:,1:k)'
Z_true 
