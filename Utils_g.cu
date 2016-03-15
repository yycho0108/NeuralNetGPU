#include "Utils.h"

extern __global__ void _sigmoid(double* d_i, double* d_o);
extern __global__ void _sigmoidPrime(double* d_i, double* d_o);
extern __global__ void _dot(double* a, double* b, double* res, int k);
extern __global__ void _mult_v(double* d_a, double* d_b, double* d_res);
extern __global__ void _add_v(double* d_a, double* d_b, double* d_res);
extern __global__ void _sub_v(double* d_a, double* d_b, double* d_res);
extern __global__ void _mult_s(double* d_v, double s, double* d_res);
extern __global__ void _add_s(double* d_v, double s, double* d_res);
extern __global__ void _sub_s(double* d_v, double s, double* d_res);
extern __global__ void _transpose(double* d_a, double* d_b);


void sigmoid_g(double* d_i, double* d_o,int n){
	_sigmoid<<<1,n>>>(d_i,d_o);
}

void sigmoidPrime_g(double* d_i, double* d_o, int n){
	_sigmoidPrime<<<1,n>>>(d_i,d_o);
}

double* dot_g(double* a, double* b, int n, int m, int k, double* res){
	if(res == nullptr){
		cudaMalloc(&res, n*m*sizeof(double));
	}
	_dot<<<n,m>>>(a,b,res,k);

	return res;
}

double* operate_v_g(double* a, double* b, int n, std::string op, bool assign){
	double* res;
	if(assign){
		res = a;
	}else{
		cudaMalloc(&res, n*sizeof(double));
	}
	if(op == "add"){
		_add_v<<<1,n>>>(a,b,res);
	}
	else if(op == "sub"){
		_sub_v<<<1,n>>>(a,b,res);
	}
	else if(op == "mult"){
		_mult_v<<<1,n>>>(a,b,res);
	}

	return res;
}
double* operate_s_g(double*& v, double s, int n, std::string op, bool assign){
	double* res;

	if(assign){
		res = v;
	}else{
		cudaMalloc(&res, n*sizeof(double));
	}

	if(op == "add"){
		_add_s<<<1,n>>>(v,s,res);
	}
	else if(op == "sub"){
		_sub_s<<<1,n>>>(v,s,res);
	}
	else if(op == "mult"){
		_mult_s<<<1,n>>>(v,s,res);
	}

	return res;
}


double* add_g(double*& a, double*& b, int n,bool assign){
	return operate_v_g(a,b,n,"add",assign);
}
double* sub_g(double*& a, double*& b, int n,bool assign){
	return operate_v_g(a,b,n,"sub",assign);
}
double* mult_g(double*& a, double*& b, int n,bool assign){
	return operate_v_g(a,b,n,"mult",assign);
}
double* add_g(double*& a, double b, int n,bool assign){
	return operate_s_g(a,b,n,"add",assign);
}
double* sub_g(double*& a, double b, int n,bool assign){
	return operate_s_g(a,b,n,"sub",assign);
}
double* mult_g(double*& a, double b,int n, bool assign){
	return operate_s_g(a,b,n,"mult",assign);
}

void transpose_g(double* src, double* dst, int n, int m){
	_transpose<<<n,m>>>(src,dst);
}

void randArr_g(int n, int m, double*& ptr){
	auto arr = randArr(n,m);
	cudaMemcpy(ptr,&arr.front(),n*m*sizeof(double),cudaMemcpyHostToDevice);
}
