#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include <omp.h>
#include <algorithm>
#include "mex.h"

using namespace std;

template<class T>
void shuffle( vector<T>& vect ){
	
	int r;
	for(int i=0;i<vect.size()-1;i++){
		r =  (rand() % (vect.size() - i-1))+1+i ;
		swap(vect[i],vect[r]);
	}
}
void change_major(double* v,int n,int m){
 //convert m*n to n*m
  double* temp = new double[n*m];
  for(int i=0;i<n*m;i++)
  	temp[i]=v[i];
  for(int i=1;i<n*m;i++) 	
  	v[(i/n)+(i%n)*m]=temp[i];

}

double pure_dot( double* v, double* x ,int numFea){
	double sum=0.0;
	for(int i=0;i<numFea;i++){
		sum += (*(v+i))*(*(x+i));
	}	
	return sum;
}

inline double get_objvalue_d(double* R, double* V, double* Z,int N,int K,int D){
	double result=0.0;
	double* temp_row= new double[D];
    for(int i=0;i<N;i++){
    	for(int j=0;j<D;j++){
    		temp_row[j]=0.0;
     	}
    	for(int j=0;j<K;j++){			
			for(int k=0;k<D;k++){
				temp_row[k]+=*(Z+i*K+j)*(*(V+j*D+k));
			}	
		} 
		for(int j=0;j<D;j++){
    		result=result+(*(R+j*N+i)-temp_row[j])*(*(R+j*N+i)-temp_row[j]);
    	}

    }
    delete[] temp_row;
    return 0.5*result;
}
void coordinate_solver(double* R,double* Z,int max_iter,double lambda, double* V, double* alpha ,double* objCD,int N,int D, int K){
   //calculate Hessian diagonal
  	double* H_bound = new double [N];
  	change_major(Z,N,K);
  	for(int i=0;i<N;i++){
		double square_sum = 0.0;
		for(int j=0;j<K;j++){
		    square_sum += (*(Z+i*K+j))*(*(Z+i*K+j));
		}
		H_bound[i]=square_sum/lambda+1.0;
	}
	//initialize 
	double* w = new double[K];
	double object_value;
    vector<int> index;
	for(int i=0;i<N;i++)
		index.push_back(i);
	int iter;
	//iteration
	for(int u=0;u<D;u++){	
	 	shuffle(index);
		iter=0;
		//initialize 
		for(int i=0;i<K;i++)
		  w[i] = 0.0;
		for(int i=0;i<N;i++)
		  *(alpha+u*N+i) = 0.0;
		object_value=0.0;
		//reach R_u
		while(iter < max_iter){
		    
			
			//start inner loop
			for(int r=0;r<N;r++){ 
				//choose random gradient to update
				int i = index[r];
				//1. compute gradient of i 
				double gi = (1.0/lambda)*pure_dot(Z+K*i,w,K)-*(R+u*N+i)+*(alpha+u*N+i);
				//2. compute alpha_u_i
				double new_alpha = *(alpha+u*N+i)-gi/H_bound[i];
				//3. maintain w
				double alpha_diff = new_alpha-*(alpha+u*N+i);
				if(  fabs(alpha_diff) > 1e-8 ){
					for(int o=0;o<K;o++){
						w[o]=max(0.0,w[o]+alpha_diff*(*(Z+K*i+o)));
					}			
					*(alpha+u*N+i) = new_alpha;
				}				
			}
			
			//if(iter%10==0)
			//computer object_value, overhead?
			//if(iter%10==0){
				double* temp = new double[K];				
				for(int i=0;i<K;i++){
					temp[i]=0.0;
				}			
				for(int i=0;i<N;i++){
					for(int j=0;j<K;j++){
						temp[j]=temp[j]+*(alpha+i+u*N)*(*(Z+i*K+j));					
					}
				}
				//take positive
				for(int j=0;j<K;j++){
						temp[j]=max(0.0,temp[j]);
				}
				object_value=(0.5/lambda)*pure_dot(temp,temp,K)-pure_dot(R+u*N,alpha+u*N,N)+0.5*pure_dot(alpha+u*N,alpha+u*N,N);
				
			    delete[] temp;
		   // }
			cout <<"D="<<u<<", iter=" << iter <<" ,object_value="<<object_value<<endl ;			
			shuffle(index);
			iter++;		
		}
		
		cerr << endl;
		//output u_th model
	//update V;
	for(int i=0;i<K;i++)
		*(V+u+i*D)=w[i];
    }
	//release memory
	// for(int i=0;i<K;i++){
	// 	for(int j=0;j<D;j++){
	// 		printf("V[%d][%d]=%lf ",i,j,*(V+i*D+j));
	// 	}
	// 	cout<<endl;
	// }
	*objCD = get_objvalue_d(R,V,Z,N,K,D);
	change_major(V,D,K);
}
void usage()
{
	mexErrMsgTxt("Usage:function [V alpha objGCD] = CDS_D (R,Z,max_iter,lambda)");
}
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	double *R,*Z,*values;
	int n,d,k;
	int max_iter;
	double lambda;
	if (nrhs!=4)
	 	usage();
	R = mxGetPr(prhs[0]);
	n= mxGetM(prhs[0]);
	d = mxGetN(prhs[0]);
	Z = mxGetPr(prhs[1]);
	k = mxGetN(prhs[1]);
	if ( (mxGetM(prhs[1]) != n) ) {
		usage();
		printf("Matrix size does not match for multiplication: R=N*D  Z=N*K");
	}
	values = mxGetPr(prhs[3]);
	lambda = values[0];
	values = mxGetPr(prhs[2]);
	max_iter = values[0];
	//V,alpha,objGCD
	plhs[0]=mxCreateDoubleMatrix(k,d,mxREAL);
    plhs[1]=mxCreateDoubleMatrix(n,d,mxREAL);
    plhs[2]=mxCreateDoubleMatrix(1,1,mxREAL);
    coordinate_solver(R,Z,max_iter,lambda,mxGetPr(plhs[0]),mxGetPr(plhs[1]),mxGetPr(plhs[2]),n,d,k);  
}
 
