#include "Parser.h"
#include "Instance.h"
#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include <omp.h>
#include <algorithm>

using namespace std;

template<class T>
void shuffle( vector<T>& vect ){
	
	int r;
	for(int i=0;i<vect.size()-1;i++){
		r =  (rand() % (vect.size() - i-1))+1+i ;
		swap(vect[i],vect[r]);
	}
}

double dot( double* v, vector<pair<int,double> >& x ){
	
	double sum=0.0;
	for(int i=0;i<x.size();i++){
	
		int index = x[i].first;
		double value = x[i].second;

		sum += v[index]*value;
	}	
	return sum;
}
double pure_dot( double* v, double* x ,int numFea){
	double sum=0.0;
	for(int i=0;i<numFea;i++){
		sum += v[i]*x[i];
	}	
	return sum;
}
inline double* get_sparse_row(vector< Instance* >* data ,int row ,int position,int numFea){
   int N = data->size();
   if(row==1){
   		double* v = new double[numFea];
    	Instance* ins = data->at(position);
    	for(int i=0;i<numFea;i++)
    		v[i]=0.0;
    	for(int i=0;i<ins->xi.size();i++){			
			int index = ins->xi[i].first;
			double value = ins->xi[i].second;
			v[index-1]=value;			
		} 		 
		return v;  	
    }   
}
inline double get_objvalue(double** R, double** V, vector< Instance* >* Z,int N,int K,int D){
	double result=0.0;
	double* temp_row= new double[D];
    for(int i=0;i<N;i++){
    	for(int j=0;j<D;j++){
    		temp_row[j]=0.0;
     	}
    	Instance* ins = Z->at(i);
    	for(int j=0;j<ins->xi.size();j++){			
			int index = ins->xi[j].first-1;
			double value = ins->xi[j].second;
			for(int k=0;k<D;k++){
				temp_row[k]+=value*V[index][k];
			}	
		} 
		for(int j=0;j<D;j++){
    		result=result+(R[j][i]-temp_row[j])*(R[j][i]-temp_row[j]);
    	}

    }
    delete[] temp_row;
    return result;
}
//index start from 1 in the data  
int main(int argc, char** argv){

	
	if( argc < 4 ){
		cerr << "Usage: dualsubsolve [Z] [R] [lambda](modelFile)\n";
		exit(0);
	}
	
	char* ZFile = argv[1];
    char* RFile = argv[2];
    double lambda = atof(argv[3]);
	char* modelFile;
	if( argc >= 5 )
		modelFile = argv[4];
	else{
		modelFile = "model";
	}
	
	int D;
	int N;
	int K;
	vector<Instance*>* data_Z =  Parser::parseSVM(ZFile,K);
	vector<Instance*>* data_R =  Parser::parseSVM(RFile,D);
	N = data_Z->size();
	cerr << "N=" << N << endl;
	cerr << "D=" << D << endl;
	cerr << "K=" << K << endl;	
	double* w = new double[K];
	double* alpha = new double[N];
	double** R = new double*[D];
	double** V =new double*[K]; 
	for(int i=0;i<K;i++){
		V[i] = new double[D];
	}
	for(int i=0;i<D;i++){
		R[i] = new double[N];
	}
	//compute upper bound and initialization
	double* H_bound = new double [N];
	for(int i=0;i<N;i++){
		double square_sum = 0.0;
		Instance* ins = data_Z->at(i);
		for(int j=0;j<ins->xi.size();j++){
		    double value = ins->xi[j].second*ins->xi[j].second;
		    square_sum += value;
		}
		H_bound[i]=square_sum/lambda+1.0;
	}
	for (int j=0;j<D;j++){
		for(int i=0;i<N;i++)
			R[j][i]=0.0;
    }
    for (int j=0;j<N;j++){
		Instance* ins = data_R->at(j);
		 for(int i=0;i<ins->xi.size();i++){			
				int index = ins->xi[i].first;
				double value = ins->xi[i].second;
				R[index-1][j]=value;					
		    }
    }
    vector<int> index;
	for(int i=0;i<N;i++)
		index.push_back(i);
	int max_iter = 100;
	int iter;
	double object_value;
	//Outer Loop D times
	ofstream fout(modelFile);
	for(int u=0;u<D;u++){		
		
		shuffle(index);
		iter=0;
		//initialize 
		for(int i=0;i<K;i++)
		  w[i] = 0.0;
		for(int i=0;i<N;i++)
		  alpha[i] = 0.0;
		object_value=0.0;
		//reach R_u
		while(iter < max_iter){			
			double update_time = -omp_get_wtime();
			//start inner loop
			for(int r=0;r<N;r++){ 
				//choose random gradient to update
				int i = index[r];
				double* Z=get_sparse_row(data_Z,1,i,K);
				//1. compute gradient of i 
				double gi = (1.0/lambda)*pure_dot(Z,w,K)-R[u][i]+alpha[i];
				//2. compute alpha_u_i
				double new_alpha = alpha[i]-gi/H_bound[i];
				//3. maintain w
				double alpha_diff = new_alpha-alpha[i];
				if(  fabs(alpha_diff) > 1e-8 ){
					for(int o=0;o<K;o++){
						w[o]=max(0.0,w[o]+alpha_diff*Z[o]);
					}			
					alpha[i] = new_alpha;
				}				
				delete[] Z;
			}
			update_time += omp_get_wtime();
			//if(iter%10==0)
			//computer object_value, overhead?
			//if(iter%10==0){
				double* temp = new double[K];				
				for(int i=0;i<K;i++){
					temp[i]=0.0;
				}			
				for(int i=0;i<N;i++){
					double* Z_temp;
					Z_temp=get_sparse_row(data_Z,1,i,K);
					for(int j=0;j<K;j++){
						temp[j]=temp[j]+alpha[i]*Z_temp[j];					
					}
					delete[] Z_temp;
				}
				//take positive
				for(int j=0;j<K;j++){
						temp[j]=max(0.0,temp[j]);
				}
				object_value=(0.5/lambda)*pure_dot(temp,temp,K)-pure_dot(R[u],alpha,N)+0.5*pure_dot(alpha,alpha,N);
				
			    delete[] temp;
		   // }
			cerr <<"D="<<u<<", iter=" << iter << ", time=" << update_time <<" ,object_value="<<object_value<<endl ;			
			shuffle(index);
			iter++;		
		}
		
		cerr << endl;
		//output u_th model
	//update V;
	for(int i=0;i<K;i++)
			V[i][u]=w[i];
	fout << u << endl;
	for(int i=0;i<N;i++)
		if( fabs(alpha[i]) > 1e-12 )
			fout << i << " " << alpha[i] << endl;
	} 	
	//release memory
	delete[] alpha;
	for(int i=0;i<K;i++){
		for(int j=0;j<D;j++){
			printf("V[%d][%d]=%lf ",i,j,V[i][j]);
		}
		cout<<endl;
	}		
    double result = get_objvalue(R,V,data_Z,N,K,D);
    cout<<"Final result:"<<result<<endl;
	fout.close();
}
