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
#endif
