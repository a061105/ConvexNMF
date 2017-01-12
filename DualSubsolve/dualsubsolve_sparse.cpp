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
	if(vect.size()==0)
	return; 
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
double pure_dot_x_nng( double* v, double* x ,int numFea){
	double sum=0.0;
	for(int i=0;i<numFea;i++){
		double xi = *(x+i);
		if( xi > 0.0 )
			sum += (*(v+i))*xi;
	}	
	return sum;
}
inline double* get_sparse_row(vector< Instance* >* data ,int position,int numFea){
   		double* v = new double[numFea];
    	Instance* ins = data->at(position);
     	for(int i=0;i<numFea;i++)
    		v[i]=0.0;
    	for(int i=0;i<ins->xi.size();i++){	
			int index = ins->xi[i].first;
			double value = ins->xi[i].second;
			v[index]=value;

		}  
		return v;  	
}
inline double get_objvalue_d(vector< Instance* >* R, double* V, vector< Instance* >* Z,int N,int K,int D){
	double result=0.0;
	double* temp_row= new double[D];
	for(int i=0;i<N;i++){
		double* r = new double[D];
		double* z = new double[K];
		z=get_sparse_row(Z,i,K);
		r=get_sparse_row(R,i,D);
		for(int j=0;j<D;j++){
			temp_row[j]=0.0;
		}
		for(int j=0;j<D;j++){			
			for(int k=0;k<K;k++){
				temp_row[j] += *(z+k)*(*(V+k*D+j));
			}	
		} 
		for(int j=0;j<D;j++){
			result=result+(*(r+j)-temp_row[j])*(*(r+j)-temp_row[j]);
		}

	}
	delete[] temp_row;
	return 0.5*result;
}
vector< Instance* >* rdata(vector<Instance*>* data, int numFea){
         
        int n = data->size();
        vector< Instance* >* reverse_data = new vector< Instance* >();
        for(int i=0;i<numFea;i++){
        	Instance* ins = new Instance();
        	reverse_data->push_back(ins);
        }
        for(int i=0;i<n;i++){
        		Instance* ins = data->at(i);
                for(int j=0;j<ins->xi.size();j++){
                	    int index = ins->xi[j].first;
                        double value = ins->xi[j].second;
                        pair<int,double> pair;                        
                        pair.first = i;
                        pair.second = value;
                        reverse_data->at(index)->xi.push_back(pair);
                }
        }

        return reverse_data;
}
void coordinate_solver(vector< Instance* >* data_R,vector< Instance* >* Z_input,int max_iter,double lambda, double* V, vector<Instance*>* A ,double& objCD,int N,int D, int K){
   vector<Instance*>* R =rdata(data_R,D);
    //calculate Hessian diagonal
  	double* H_bound = new double [N];
	for(int i=0;i<N;i++){
		Instance* ins = Z_input->at(i);
		double square_sum = 0.0;
		for(int j=0;j<ins->xi.size();j++){
		    square_sum += ins->xi[j].second*ins->xi[j].second;
		}
		H_bound[i]=square_sum/lambda+1.0;
	}
	//initialize 
	double* w = new double[K];	
	double object_value;
	int iter;
	int nnz;
	//iteration
	for(int u=0;u<D;u++){
        //create sparse referense
        nnz = R->at(u)->xi.size();
        int* table = new int[nnz];
        double* r_value = new double[nnz];
        for(int j=0;j<nnz;j++){
		    table[j]=R->at(u)->xi[j].first;
		    r_value[j]=R->at(u)->xi[j].second;
		}
		//random shuffle
		vector<int> index;
		for(int i=0;i<nnz;i++)
			index.push_back(i);
		shuffle(index);
		iter=0;
		//initialize 		
		for(int i=0;i<K;i++)
			w[i] = 0.0;
		double* alpha = new double[nnz];
		for(int i=0;i<nnz;i++)
			alpha[i] = 0.0;
		object_value=0.0;
		while(iter < max_iter){
			
			double update_time = -omp_get_wtime();
			//start inner loop
			for(int r=0;r<nnz;r++){ 
				//choose random gradient to update
				int i = index[r];
				double* Z=get_sparse_row(Z_input,table[i],K);
				//1. compute gradient of i 
				double gi = (1.0/lambda)*pure_dot_x_nng(Z,w,K)-r_value[i]+alpha[i];
				//2. compute alpha_u_i
		     	double new_alpha = alpha[i]-gi/H_bound[table[i]];
				//3. maintain w
				double alpha_diff = new_alpha-alpha[i];
				if(  fabs(alpha_diff) > 1e-8 ){
					for(int o=0;o<K;o++){
						w[o] = w[o]+alpha_diff*(*(Z+o));
					}			
					alpha[i] = new_alpha;
				}
				delete [] Z;				
			}
			update_time += omp_get_wtime();
			//test
			double* temp = new double[K];	
				for(int i=0;i<K;i++){
					temp[i]=0.0;
				}			
				for(int i=0;i<nnz;i++){
					double* Z_temp=get_sparse_row(Z_input,table[i],K);
					for(int j=0;j<K;j++){
						temp[j]=temp[j]+alpha[i]*Z_temp[j];					
					}
					delete[] Z_temp;
				}
				//take positive
				for(int j=0;j<K;j++){
						temp[j]=max(0.0,temp[j]);
				}
				object_value=(0.5/lambda)*pure_dot(temp,temp,K)-pure_dot(r_value,alpha,nnz)+0.5*pure_dot(alpha,alpha,nnz);
				
			    delete[] temp;
		  
			cerr <<"D="<<u<<", iter=" << iter << ", time=" << update_time <<" ,object_value="<<object_value<<endl ;			
			// obj decrease
			shuffle(index);     		
			iter++;
		}
		for(int i=0;i<K;i++)
			*(V+i+u*K)= max(0.0, w[i]);
        for(int i=0;i<nnz;i++){
                        pair<int,double> pair;                        
                        pair.first = table[i];
                        pair.second = r_value[i];
                        A->at(u)->xi.push_back(pair);            
		 }			
		index.clear();
		delete[] alpha;
	}
	objCD = get_objvalue_d(data_R,V,Z_input,N,K,D);
	delete[] w;	
}
//index start at 1
int main(int argc, char** argv){

	if( argc < 5 ){
		cerr << "Usage: ./dualsubsolve_sparse [Z] [R] [lambda] [max_iter] (modelFile)\n";
		exit(0);
	}
	
	char* ZFile = argv[1];
    char* RFile = argv[2];
    double lambda = atof(argv[3]);
    int  max_iter=atoi(argv[4]);
	char* modelFile;
	if( argc >= 6 )
		modelFile = argv[5];
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
	vector<Instance*>* A = new vector< Instance* >();
	for(int i=0;i<D;i++){
        	Instance* ins = new Instance();
        	A->push_back(ins);
    }
	double* V = new double[K*D];
	double objCD;	
	coordinate_solver(data_R,data_Z,max_iter,lambda,V,A,objCD,N,D,K);  
    cout<<"Final result:"<<objCD<<endl;
}