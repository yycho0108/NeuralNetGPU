
#include "Utils.h"

__global__ void _sigmoid(double* d_i, double* d_o){
	int idx = threadIdx.x;
	d_o[idx] = 1.0 / (1.0 + exp(-d_i[idx]));
}

void sigmoid(std::vector<double>& i, std::vector<double>& o){
	unsigned int n = i.size();
	double *d_i, *d_o;

	cudaMalloc(&d_i, n*sizeof(double));
	cudaMalloc(&d_o, n*sizeof(double));

	cudaMemcpy(d_i,&i.front(),n*sizeof(double),cudaMemcpyHostToDevice);

	_sigmoid<<<1,n>>>(d_i,d_o);

	cudaMemcpy(&o.front(),d_o,n*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(d_i);
	cudaFree(d_o);

}

__global__ void _sigmoidPrime(double* d_i, double* d_o){
	int idx = threadIdx.x;
	double ex = exp(-d_i[idx]);
	d_o[idx] = ex/((1+ex)*(1+ex));
}

std::vector<double> sigmoidPrime(std::vector<double>& v){
	unsigned int n = v.size();
	double *d_i, *d_o;

	cudaMalloc(&d_i, n*sizeof(double));
	cudaMalloc(&d_o, n*sizeof(double));

	cudaMemcpy(d_i,&v.front(),n*sizeof(double),cudaMemcpyHostToDevice);

	_sigmoidPrime<<<1,n>>>(d_i,d_o);
	std::vector<double> res;
	cudaMemcpy(&res.front(),d_o,n*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(d_i);
	cudaFree(d_o);
	return res;
}

__global__ void _dot(double* a, double* b, double* res, int k){
	unsigned int i = blockIdx.x;
	unsigned int j = threadIdx.x;

	//unsigned int n = gridDim.x;	 //# blocks / grid
	unsigned int m = blockDim.x; //# threads / block

	res[i*m+j] = 0;
	for(int l=0;l<k;++l){
		res[i*m+j] += a[i*k+l] * b[l*m+j]; //a[i] dot b[j]
	}
	return;
}
std::vector<double> dot(std::vector<double>& a, std::vector<double>& b, int k){

	int n = a.size() / k; //a = n*k
	int m = b.size() / k; //b = k*m


	/*std::cout << '|';
	for(auto& i : a){
			std::cout << i << ' ';
	}
	std::cout << '|';
	for(auto& i : b){
			std::cout << i << ' ';
	}
	std::cout << '|';*/


	//printf("[a=%d,b=%d,k=%d,n=%d,m=%d]",a.size(),b.size(),k,n,m);

	double *d_a, *d_b, *d_res;
	cudaMalloc(&d_a, n*k*sizeof(double));
	cudaMalloc(&d_b, k*m*sizeof(double));
	cudaMalloc(&d_res, n*m*sizeof(double));

	std::vector<double> res(n*m);
	cudaMemcpy(d_a,&a.front(),n*k*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,&b.front(),k*m*sizeof(double),cudaMemcpyHostToDevice);
	_dot<<<n,m>>>(d_a,d_b,d_res,k);

	cudaMemcpy(&res.front(),d_res,n*m*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);

	return res;
}
__global__ void _mult_v(double* d_a, double* d_b, double* d_res){
	int idx = threadIdx.x;
	d_res[idx] = d_a[idx] * d_b[idx];
}
__global__ void _add_v(double* d_a, double* d_b, double* d_res){
	int idx = threadIdx.x;
	d_res[idx] = d_a[idx] + d_b[idx];
}
__global__ void _sub_v(double* d_a, double* d_b, double* d_res){
	int idx = threadIdx.x;
	d_res[idx] = d_a[idx] - d_b[idx];
}

__global__ void _mult_s(double* d_v, double s, double* d_res){
	int idx = threadIdx.x;
	d_res[idx] = d_v[idx] * s;
}
__global__ void _add_s(double* d_v, double s, double* d_res){
	int idx = threadIdx.x;
	d_res[idx] = d_v[idx] + s;
}
__global__ void _sub_s(double* d_v, double s, double* d_res){
	int idx = threadIdx.x;
	d_res[idx] = d_v[idx] - s;
}


std::vector<double> operate_v(std::vector<double>& a, std::vector<double>& b, std::string op){
	int n = a.size();
	double *d_a, *d_b;
	cudaMalloc(&d_a, n*sizeof(double));
	cudaMalloc(&d_b, n*sizeof(double));
	cudaMemcpy(d_a,&a.front(),n*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,&b.front(),n*sizeof(double),cudaMemcpyHostToDevice);

	if(op == "add"){
		_add_v<<<1,n>>>(d_a,d_b,d_a);
	}
	else if(op == "sub"){
		_sub_v<<<1,n>>>(d_a,d_b,d_a);
	}
	else if(op == "mult"){
		_mult_v<<<1,n>>>(d_a,d_b,d_a);
	}

	std::vector<double> res(n);

	cudaMemcpy(&res.front(),d_a,n*sizeof(double),cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);

	return res;
}
std::vector<double> operate_s(std::vector<double>& v, double s, std::string op){
	int n = v.size();
	double *d_v;
	cudaMalloc(&d_v, n*sizeof(double));
	cudaMemcpy(d_v,&v.front(),n*sizeof(double),cudaMemcpyHostToDevice);

	if(op == "add"){
		_add_s<<<1,n>>>(d_v,s,d_v);
	}
	else if(op == "sub"){
		_sub_s<<<1,n>>>(d_v,s,d_v);
	}
	else if(op == "mult"){
		_mult_s<<<1,n>>>(d_v,s,d_v);
	}

	std::vector<double> res(n);
	cudaMemcpy(&res.front(),d_v,n*sizeof(double),cudaMemcpyDeviceToHost);
	cudaFree(d_v);

	return res;
}


std::vector<double> add(std::vector<double>& a, std::vector<double>& b){
	return operate_v(a,b,"add");
}
std::vector<double> sub(std::vector<double>& a, std::vector<double>& b){
	return operate_v(a,b,"sub");
}
std::vector<double> mult(std::vector<double>& a, std::vector<double>& b){
	return operate_v(a,b,"mult");
}
std::vector<double> add(std::vector<double>& a, double b){
	return operate_s(a,b,"add");
}
std::vector<double> sub(std::vector<double>& a, double b){
	return operate_s(a,b,"sub");
}
std::vector<double> mult(std::vector<double>& a, double b){
	return operate_s(a,b,"mult");
}

__global__ void _transpose(double* d_a, double* d_b){
	int n = gridDim.x;
	int m = blockDim.x;

	int i = blockIdx.x;
	int j = threadIdx.x;
	d_b[j*n+i] = d_a[i*m + j];
}

void transpose(std::vector<double>& src, std::vector<double>& dst, int n, int m){
	double *d_src, *d_dst;

	cudaMalloc(&d_src, n*m*sizeof(double));
	cudaMalloc(&d_dst, n*m*sizeof(double));

	cudaMemcpy(d_src,&src.front(),n*m*sizeof(double),cudaMemcpyHostToDevice);
	_transpose<<<n,m>>>(d_src,d_dst);
	cudaMemcpy(&dst.front(),d_dst,n*m*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(d_src);
	cudaFree(d_dst);

}

double randNum(){
	static auto _randNum = std::bind(std::uniform_real_distribution<double>(0.0,1.0),std::default_random_engine(time(0))); //random
	return _randNum();
}
std::vector<double> randArr(int n, int m){
	std::vector<double> res(n*m);
	std::generate(res.begin(),res.end(),randNum);
	return res;
}

void XOR_GEN(std::vector<double>& X, std::vector<double>& Y){
	X[0] = randNum()>0.5?1:0;
	X[1] = randNum()>0.5?1:0;
	Y[0] = int(X[0]) ^ int(X[1]);
}
