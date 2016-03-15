#ifndef __UTILS__H__
#define __UTILS__H__


#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <functional>
#include <ctime>

extern std::vector<double> dot(std::vector<double>& a, std::vector<double>& b, int k);
extern std::vector<double> mult(std::vector<double>& a, std::vector<double>& b);
extern std::vector<double> sub(std::vector<double>& a, std::vector<double>& b);
extern std::vector<double> add(std::vector<double>& a, std::vector<double>& b);
extern std::vector<double> mult(std::vector<double>& a, double b);

extern void transpose(std::vector<double>& src, std::vector<double>& dst, int n, int m);
extern std::vector<double> sigmoidPrime(std::vector<double>& v);
extern std::vector<double> randArr(int n, int m);

extern double randNum();
extern void XOR_GEN(std::vector<double>& X, std::vector<double>& Y);


extern void randArr_g(int n, int m, double*& ptr);

extern void sigmoid_g(double*,double*,int);
extern void sigmoidPrime_g(double*,double*,int);
extern void transpose_g(double*,double*,int,int);
extern double* dot_g(double* a, double* b, int n, int m, int k, double* res = nullptr);

extern double* add_g(double*& a, double*& b,int n, bool assign=false);
extern double* sub_g(double*& a, double*& b,int n, bool assign=false);
extern double* mult_g(double*& a, double*& b,int n, bool assign=false);

extern double* add_g(double*& a, double b,int n, bool assign=false);
extern double* sub_g(double*& a, double b,int n, bool assign=false);
extern double* mult_g(double*& a, double b,int n, bool assign=false);

#endif
