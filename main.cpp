#include "Net.h"
#include <cuda.h>
#include <iostream>

/*
 * extern std::vector<double> dot(std::vector<double>& a, std::vector<double>& b, int k);
 * void testDot(){
	std::vector<double> a({2,3,5});
	std::vector<double> b({7,11,13});

	std::vector<double> res = dot(a,b,3);


	for(auto& i : res){
		std::cout << i << ' ';
	}
	std::cout << std::endl;
}*/

/*
 * extern void transpose(std::vector<double>& src, std::vector<double>& dst, int n, int m);
 *
 * void testTranspose(){
	std::vector<double> src({1,2,3,4,5,6});
	std::vector<double> dst(6);
	transpose(src,dst,3,2);
	for(int i=0;i<6;++i){
		std::cout << dst[i] << ' ';
	}
}*/

int main(){
	std::vector<int> topology({2,4,1});
	auto net = Net(topology);
	std::vector<double> X(2);
	std::vector<double> Y(1);

	for(int i=0;i<2000;++i){
		XOR_GEN(X,Y);
		net.BP(X,Y);
	}
	for(int i=0;i<10;++i){
		XOR_GEN(X,Y);
		std::cout << net.FF(X)[0] << ',' << Y[0] << std::endl;
	}
	return 0;
}
